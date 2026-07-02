"""Export actor plus the complete deployment contract derived from training cfg."""
from pathlib import Path
import argparse, importlib, json, os, sys, xml.etree.ElementTree as ET
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
# Isaac Gym must initialize its binary bindings before torch in this project.
import isaacgym
import torch

class Actor(torch.nn.Sequential):
    def __init__(self, observations):
        super().__init__(torch.nn.Linear(observations,512),torch.nn.ELU(),torch.nn.Linear(512,256),
                         torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),
                         torch.nn.Linear(128,12))


class NegativeLateralMirror(torch.nn.Module):
    """Use the learned +vy behavior as an exact sagittal mirror for -vy."""
    def __init__(self, actor, command_lin_scale):
        super().__init__(); self.actor=actor
        self.command_lin_scale=float(command_lin_scale)
        obs_index=list(range(52));obs_sign=[1.0]*52
        for index in (1,3,5,7,10,11,48,49,50):obs_sign[index]=-1.0
        leg_map=(1,0,3,2)
        for start in (12,24,36):
            for dst_leg,src_leg in enumerate(leg_map):
                for joint in range(3):
                    obs_index[start+dst_leg*3+joint]=start+src_leg*3+joint
                obs_sign[start+dst_leg*3]=-1.0
        out_index=list(range(12));out_sign=[1.0]*12
        for dst_leg,src_leg in enumerate(leg_map):
            for joint in range(3):out_index[dst_leg*3+joint]=src_leg*3+joint
            out_sign[dst_leg*3]=-1.0
        self.register_buffer("obs_index",torch.tensor(obs_index,dtype=torch.long))
        self.register_buffer("obs_sign",torch.tensor(obs_sign))
        self.register_buffer("out_index",torch.tensor(out_index,dtype=torch.long))
        self.register_buffer("out_sign",torch.tensor(out_sign))

    def forward(self,observations):
        adjusted=observations.clone()
        vx=observations[:,9:10]/self.command_lin_scale
        vy=observations[:,10:11]/self.command_lin_scale
        pure_lateral=torch.exp(-torch.square(vx/0.03))
        lateral_activity=1.0-torch.exp(-torch.square(vy/0.02))
        compensation=torch.clamp(0.08-0.45*torch.abs(vy),min=0.03,max=0.07)
        adjusted[:,9:10]=adjusted[:,9:10]+compensation*self.command_lin_scale*pure_lateral*lateral_activity
        native=self.actor(adjusted)
        mirrored_obs=adjusted.index_select(1,self.obs_index)*self.obs_sign
        mirrored=self.actor(mirrored_obs)
        mirrored=mirrored.index_select(1,self.out_index)*self.out_sign
        return torch.where(observations[:,10:11] < 0.0,mirrored,native)

def matched(mapping,name):
    values=[value for key,value in mapping.items() if key in name]
    if len(values)!=1: raise ValueError(f"Expected one cfg match for {name}, got {values}")
    return values[0]

def deployment_config(cfg, checkpoint, gym_root):
    names=list(cfg.control.policy_joint_order)
    urdf_path=Path(cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}",str(gym_root)))
    root=ET.parse(urdf_path).getroot()
    effort={j.get("name"):float(j.find("limit").get("effort")) for j in root.findall("joint") if j.find("limit") is not None}
    scales=[]
    for name in names:
        if "hip" in name: scales.append(cfg.control.hip_action_scale)
        elif name.startswith(("RL_","RR_")): scales.append(cfg.control.rear_action_scale)
        else: scales.append(cfg.control.action_scale)
    return {
        "schema_version":1,"task":cfg.__name__,"checkpoint":str(checkpoint.resolve()),
        "dimensions":{"observations":cfg.env.num_observations,"actions":cfg.env.num_actions},
        "joint_names":names,
        "default_joint_angles":[cfg.init_state.default_joint_angles[n] for n in names],
        "initial_state":{"base_position":list(cfg.init_state.pos),"base_quaternion_xyzw":list(cfg.init_state.rot)},
        "control":{"sim_dt":cfg.sim.dt,"decimation":cfg.control.decimation,
                   "stiffness":[matched(cfg.control.stiffness,n) for n in names],
                   "damping":[matched(cfg.control.damping,n) for n in names],
                   "action_scale":scales,"torque_limits":[effort[n] for n in names],
                   "output_transform":"tanh"},
        "observations":{"clip":cfg.normalization.clip_observations,
                        "lin_vel_scale":cfg.normalization.obs_scales.lin_vel,
                        "ang_vel_scale":cfg.normalization.obs_scales.ang_vel,
                        "dof_pos_scale":cfg.normalization.obs_scales.dof_pos,
                        "dof_vel_scale":cfg.normalization.obs_scales.dof_vel,
                        "command_scale":[cfg.normalization.obs_scales.lin_vel,cfg.normalization.obs_scales.lin_vel,cfg.normalization.obs_scales.ang_vel],
                        "layout":["base_lin_vel","base_ang_vel","projected_gravity","commands","dof_pos_error","dof_vel","previous_actions","gait_phase_sin_cos","heading_error_sin_cos"]},
        "commands":{"default":[0.0,0.0,0.0],"heading_command":cfg.commands.heading_command,
                    "observe_heading_error":getattr(cfg.commands,"observe_heading_error",False),
                    "ranges":{"lin_vel_x":list(cfg.commands.ranges.lin_vel_x),"lin_vel_y":list(cfg.commands.ranges.lin_vel_y),"ang_vel_yaw":list(cfg.commands.ranges.ang_vel_yaw)},
                    "default_heading":sum(cfg.commands.ranges.heading)/2 if cfg.commands.heading_command else 0.0,"heading_gain":0.5},
        "gait":{"period":cfg.rewards.gait_period,"stance_ratio":cfg.rewards.gait_stance_ratio,
                "thigh_amplitude":cfg.rewards.gait_thigh_amplitude,"calf_amplitude":cfg.rewards.gait_calf_amplitude,
                "phase_offsets":{"FL":0.0,"FR":0.5,"RL":0.5,"RR":0.0}},
        "episode_length_s":cfg.env.episode_length_s,
    }

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--config-class",default="legged_gym.envs.fanfan.fanfan_config:FanfanRoughCfg")
    p.add_argument("--mirror-negative-lateral",action="store_true")
    p.add_argument("--gym-root",type=Path,default=Path(__file__).resolve().parents[1]/"unitree_rl_gym");a=p.parse_args()
    sys.path.insert(0,str(a.gym_root));module_name,class_name=a.config_class.split(":",1);cfg=getattr(importlib.import_module(module_name),class_name)
    state=torch.load(a.checkpoint,map_location="cpu")["model_state_dict"]
    actor=Actor(cfg.env.num_observations).eval();actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("actor.")})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    export_actor=NegativeLateralMirror(actor,cfg.normalization.obs_scales.lin_vel).eval() if a.mirror_negative_lateral else actor
    torch.onnx.export(export_actor,torch.zeros(1,cfg.env.num_observations),a.output,input_names=["observations"],output_names=["raw_actions"],dynamic_axes={"observations":{0:"batch"},"raw_actions":{0:"batch"}},opset_version=17)
    manifest=deployment_config(cfg,a.checkpoint,a.gym_root)
    manifest["negative_lateral_mirroring"]=a.mirror_negative_lateral
    manifest["pure_lateral_vx_compensation"]="clip(0.08-0.45*abs(vy),0.03,0.07)" if a.mirror_negative_lateral else 0.0
    import onnx
    model=onnx.load(a.output);entry=model.metadata_props.add();entry.key="fanfan_deployment_config";entry.value=json.dumps(manifest,separators=(",",":"));onnx.save(model,a.output)
    sidecar=a.output.with_suffix(".json");sidecar.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Exported {a.output} with cfg metadata and {sidecar}")
