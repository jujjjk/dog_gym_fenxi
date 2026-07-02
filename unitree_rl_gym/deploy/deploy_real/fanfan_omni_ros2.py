#!/usr/bin/env python3
"""ROS2 ONNX policy/safety supervisor for fanfan.

Publishes safe joint targets; a robot-specific RS01 bus adapter must consume
`/fanfan/policy_joint_targets`. It intentionally does not guess motor IDs or a
vendor transport.
"""
import argparse, csv, json, math, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, String


def quat_matrix(x, y, z, w):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


class FanfanOmniNode(Node):
    def __init__(self, policy_path, csv_path):
        super().__init__("fanfan_omni_policy")
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.cfg = json.loads(metadata["fanfan_deployment_config"])
        if self.cfg["dimensions"] != {"observations": 52, "actions": 12}:
            raise RuntimeError(f"Expected 52x12 omni policy, got {self.cfg['dimensions']}")
        c, ctl, obs = self.cfg, self.cfg["control"], self.cfg["observations"]
        self.names = c["joint_names"]
        self.default = np.asarray(c["default_joint_angles"], np.float32)
        self.kp = np.asarray(ctl["stiffness"], np.float32)
        self.kd = np.asarray(ctl["damping"], np.float32)
        self.scale = np.asarray(ctl["action_scale"], np.float32)
        self.torque_limits = np.asarray(ctl["torque_limits"], np.float32)
        self.obs_cfg, self.gait = obs, c["gait"]
        self.dt = float(ctl["sim_dt"] * ctl["decimation"])

        self.desired_cmd = np.zeros(3, np.float32)
        self.cmd = np.zeros(3, np.float32)
        self.cmd_min = np.array([-0.06, -0.04, -0.35], np.float32)
        self.cmd_max = np.array([0.18, 0.04, 0.35], np.float32)
        self.cmd_step = np.array([0.01, 0.005, 0.03], np.float32)
        self.action = np.zeros(12, np.float32)
        self.target = self.default.copy()
        self.q = self.dq = self.quat = self.omega = self.velocity = None
        self.position = None
        self.path_origin = None
        self.path_heading = 0.0
        self.cross_integral = 0.0
        self.last_joint = self.last_imu = self.last_odom = self.last_cmd = 0.0
        self.stand_ready = False
        self.estop_latched = False
        self.counter = 0
        self.heading_target = None
        self.target_step_limit = 0.03
        self.q_error_limit = 0.30
        self.roll_limit, self.pitch_limit = 0.45, 0.35
        self.feedback_timeout, self.cmd_timeout = 0.10, 0.25

        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint, 20)
        self.create_subscription(Imu, "/imu/data", self.on_imu, 20)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Bool, "/fanfan/stand_ready", self.on_ready, 10)
        self.create_subscription(Bool, "/fanfan/emergency_stop", self.on_estop, 10)
        self.target_pub = self.create_publisher(JointState, "/fanfan/policy_joint_targets", 10)
        self.enabled_pub = self.create_publisher(Bool, "/fanfan/policy_enabled", 10)
        self.status_pub = self.create_publisher(String, "/fanfan/policy_status", 10)
        self.timer = self.create_timer(self.dt, self.tick)

        self.csv_file = open(csv_path, "w", newline="") if csv_path else None
        self.csv = csv.writer(self.csv_file) if self.csv_file else None
        if self.csv:
            self.csv.writerow(["time","vx_cmd","vy_cmd","yaw_cmd","vx","vy","vz","roll","pitch",
                "max_q_error","max_raw_torque","inference_ms","joint_age_ms","imu_age_ms","odom_age_ms","enabled"]
                +[f"q_{n}" for n in self.names]+[f"dq_{n}" for n in self.names]
                +[f"target_{n}" for n in self.names]+[f"raw_torque_{n}" for n in self.names])

    def on_cmd(self, msg):
        self.desired_cmd[:] = np.clip([msg.linear.x,msg.linear.y,msg.angular.z],self.cmd_min,self.cmd_max)
        self.last_cmd = time.monotonic()

    def on_joint(self, msg):
        index = {name:i for i,name in enumerate(msg.name)}
        if not all(name in index for name in self.names): return
        self.q = np.array([msg.position[index[n]] for n in self.names],np.float32)
        self.dq = np.array([msg.velocity[index[n]] for n in self.names],np.float32)
        self.last_joint = time.monotonic()

    def on_imu(self, msg):
        q=msg.orientation;self.quat=np.array([q.x,q.y,q.z,q.w],np.float32)
        w=msg.angular_velocity;self.omega=np.array([w.x,w.y,w.z],np.float32)
        self.last_imu=time.monotonic()

    def on_odom(self, msg):
        v=msg.twist.twist.linear;self.velocity=np.array([v.x,v.y,v.z],np.float32)
        p=msg.pose.pose.position;self.position=np.array([p.x,p.y],np.float32)
        self.last_odom=time.monotonic()

    def on_ready(self, msg): self.stand_ready=bool(msg.data)
    def on_estop(self, msg):
        if msg.data: self.estop_latched=True

    def gait_offset(self, phase):
        out=np.zeros(12,np.float32);ratio=self.gait["stance_ratio"]
        for i,name in enumerate(self.names):
            p=(phase+self.gait["phase_offsets"][name[:2]])%1.0
            s=np.clip((p-ratio)/(1-ratio),0,1);smooth=s*s*(3-2*s)
            if "thigh" in name:
                out[i]=self.gait["thigh_amplitude"]*((-1+2*np.clip(p/ratio,0,1)) if p<ratio else (1-2*smooth))
            elif "calf" in name:
                out[i]=self.gait["calf_amplitude"]*np.sin(np.pi*smooth)*(p>=ratio)
        return out

    def disable(self, reason):
        self.enabled_pub.publish(Bool(data=False));self.status_pub.publish(String(data=reason))

    def tick(self):
        now=time.monotonic()
        feedback_ok=(self.q is not None and self.quat is not None and self.velocity is not None
                     and now-min(self.last_joint,self.last_imu,self.last_odom)<self.feedback_timeout
                     and min(self.last_joint,self.last_imu,self.last_odom)>0)
        if now-self.last_cmd>self.cmd_timeout:self.desired_cmd.fill(0)
        if self.estop_latched or not self.stand_ready or not feedback_ok:
            self.heading_target = None
            self.path_origin = None;self.cross_integral = 0.0
            self.disable("estop" if self.estop_latched else "not_ready_or_feedback_stale");return

        R=quat_matrix(*self.quat);roll=math.atan2(R[2,1],R[2,2]);pitch=math.asin(np.clip(-R[2,0],-1,1));yaw=math.atan2(R[1,0],R[0,0])
        if abs(roll)>self.roll_limit or abs(pitch)>self.pitch_limit:
            self.estop_latched=True;self.disable("attitude_guard");return
        self.cmd += np.clip(self.desired_cmd-self.cmd,-self.cmd_step,self.cmd_step)
        if self.heading_target is None:self.heading_target=yaw
        self.heading_target=math.atan2(math.sin(self.heading_target+self.cmd[2]*self.dt),math.cos(self.heading_target+self.cmd[2]*self.dt))
        heading_error=math.atan2(math.sin(self.heading_target-yaw),math.cos(self.heading_target-yaw))
        policy_cmd=self.cmd.copy()
        straight_mode=(self.cmd[0]>0.03 and abs(self.cmd[1])<0.005 and abs(self.cmd[2])<0.05)
        if straight_mode and self.position is not None:
            if self.path_origin is None:
                self.path_origin=self.position.copy();self.path_heading=yaw;self.cross_integral=0.0
            normal=np.array([-math.sin(self.path_heading),math.cos(self.path_heading)],np.float32)
            cross=float(np.dot(self.position-self.path_origin,normal));self.cross_integral=float(np.clip(self.cross_integral+cross*self.dt,-0.5,0.5))
            policy_cmd[1]=np.clip(-0.005-0.20*cross-0.03*self.cross_integral-0.10*self.velocity[1],self.cmd_min[1],self.cmd_max[1])
        else:
            self.path_origin=None;self.cross_integral=0.0
        gravity=R.T@np.array([0,0,-1],np.float32)
        phase=(self.counter*self.dt%self.gait["period"])/self.gait["period"]
        o=self.obs_cfg
        obs=np.concatenate((self.velocity*o["lin_vel_scale"],self.omega*o["ang_vel_scale"],gravity,
            policy_cmd*np.asarray(o["command_scale"]),(self.q-self.default)*o["dof_pos_scale"],
            self.dq*o["dof_vel_scale"],self.action,[np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),np.sin(heading_error),np.cos(heading_error)])).astype(np.float32)
        inference_start=time.perf_counter()
        raw=self.session.run(["raw_actions"],{"observations":obs[None]})[0][0]
        inference_ms=(time.perf_counter()-inference_start)*1000.0
        self.action=np.tanh(raw).astype(np.float32)
        requested=self.default+self.scale*self.action+self.gait_offset(phase)
        target=self.target+np.clip(requested-self.target,-self.target_step_limit,self.target_step_limit)
        q_error=np.abs(target-self.q);raw_torque=self.kp*(target-self.q)-self.kd*self.dq
        if q_error.max()>self.q_error_limit or np.any(np.abs(raw_torque)>self.torque_limits):
            self.estop_latched=True;self.disable("q_error_or_torque_guard");return
        self.target=target;self.counter+=1
        msg=JointState();msg.header.stamp=self.get_clock().now().to_msg();msg.name=self.names
        msg.position=self.target.astype(float).tolist();msg.velocity=[0.0]*12;msg.effort=raw_torque.astype(float).tolist()
        self.target_pub.publish(msg);self.enabled_pub.publish(Bool(data=True))
        if self.csv:
            self.csv.writerow([time.time(),*self.cmd,*self.velocity,roll,pitch,q_error.max(),np.abs(raw_torque).max(),inference_ms,
                (now-self.last_joint)*1000,(now-self.last_imu)*1000,(now-self.last_odom)*1000,1,
                *self.q,*self.dq,*self.target,*raw_torque]);self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file:self.csv_file.close()
        super().destroy_node()


def main():
    p=argparse.ArgumentParser();p.add_argument("policy");p.add_argument("--csv",default="fanfan_omni.csv");args=p.parse_args()
    rclpy.init();node=FanfanOmniNode(args.policy,args.csv)
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()

if __name__=="__main__":main()
