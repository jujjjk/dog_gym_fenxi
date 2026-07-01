#!/usr/bin/env python3
"""Run the V4 command acceptance matrix and write a compact CSV report."""
from pathlib import Path
import argparse, csv
import mujoco, numpy as np
from sim2sim import Sim, body_linear_velocity

CASES = {
    "forward": (0.20, 0.00, 0.00), "backward": (-0.08, 0.00, 0.00),
    "left_strafe": (0.00, 0.05, 0.00), "right_strafe": (0.00, -0.05, 0.00),
    "turn_left": (0.00, 0.00, 0.40), "turn_right": (0.00, 0.00, -0.40),
    "forward_left": (0.15, 0.00, 0.30), "forward_right": (0.15, 0.00, -0.30),
    "backward_turn": (-0.08, 0.00, 0.20), "diagonal": (0.12, 0.04, 0.00),
}

def evaluate(model, policy, command, duration):
    s=Sim(model,policy,command);start=s.d.qpos[:3].copy();raw=[];vel=[];min_z=99
    for i in range(round(duration/s.m.opt.timestep)):
        if i%s.decimation==0:s.policy()
        raw.append(s.step());vel.append(body_linear_velocity(s.d.qpos[3:7],s.d.qvel[:3]));min_z=min(min_z,s.d.qpos[2])
    q=s.d.qpos[3:7];R=np.empty(9);mujoco.mju_quat2Mat(R,q);R=R.reshape(3,3);yaw=np.arctan2(R[1,0],R[0,0]);raw=np.abs(raw);vel=np.asarray(vel)
    return [*(s.d.qpos[:3]-start),yaw,*vel.mean(0)[:2],min_z,raw.mean(),raw.max(),(raw>s.limits).mean()]

def main():
    root=Path(__file__).parent;p=argparse.ArgumentParser();p.add_argument("--policy",type=Path,default=root/"models/fanfan_omni.onnx");p.add_argument("--model",type=Path,default=root/"models/fanfan_scene.xml");p.add_argument("--duration",type=float,default=20);p.add_argument("--output",type=Path,default=root/"omni_matrix.csv");a=p.parse_args()
    with a.output.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["case","vx_cmd","vy_cmd","yaw_cmd","dx","dy","dz","yaw","vx_mean","vy_mean","min_z","torque_mean","torque_max","over_limit"])
        for name,cmd in CASES.items():w.writerow([name,*cmd,*evaluate(a.model,a.policy,cmd,a.duration)])
    print(a.output)
if __name__=="__main__":main()
