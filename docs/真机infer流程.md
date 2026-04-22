参考：[松灵真机infer流程](https://jwolpxeehx.feishu.cn/wiki/U5dgwsBiRiwxpakOIgdc0ptrneh)，[策略系统测试案例](https://jwolpxeehx.feishu.cn/wiki/Vabfwt2FXiAOIGksmmIc6gH3nSe)

# 电脑（翠湖）

1. ssh -g -L 5001:localhost:5001 -CAXY (your job) -p 2222
2. (tmux)
3. 
   1. ```Bash
      # 启用环境
      source /share/project/wujiling/openpi/.venv/bin/activate
      
      # 退出 conda，避免 PATH 干扰
      conda deactivate || true
      
      # 替换 ROOT
      export ROOT=/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
      export PYTHONHOME="$ROOT"
      
      # 启动 server
      python /share/project/wujiling/openpi/examples/test/serve_robobrain_robotics_pi0.py
      ```

# 机器：

1. （跟电脑连同一个wifi，改ip，能ping通）
2. bash ~/info.sh （不同机器不同，可以参考Destop下的record.sh，主要是一个can配置一个是几个python包环境）
3. Conda activate dora
4. 机器本地启动脚本：/home/agilex/robobrain-robotics/client_move_from_server_eef.py

注意：

1. eef和qpos区别，二者都是约定先右后左，包括发送的action和get的state
2. infer和replay端口的区别
3. 注意运行环境
4. 注意修改model的path
5. 出问题可以根据报错修改，测试运动可以单独运行test_ctrl文件。



样例脚本：

1.服务端：

```
**文件名：serve_robobrain_robotics_.py**     
"""
POST /infer 输入样例：
{
  "qpos": [[0.1, 0.2, ..., 0.3]],  # shape: [B, state_dim]，可为一维或二维数组
  "eef_pose": [[0.1, 0.2, ..., 0.3]],  # shape: [B, action_dim]，可为一维或二维数组
  "instruction": "请让机器人前进并避开障碍物",
  "images": [
    {
      "base_0_rgb": "<base64字符串>",
      "left_wrist_0_rgb": "<base64字符串>"
    }
    # 可以有多个样本，每个样本是一个相机名到base64图片的字典
  ],
}
"""

"""huaihai path:
/share/project/lvhuaihai# conda activate envs/openpi/
"""
import os
import io
import base64
from PIL import Image
import sys
import torch
import h5py
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import traceback
from torchvision import transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 启用CORS，允许跨域请求
CORS(app)

# 服务配置
SERVICE_CONFIG = {
    'host': '0.0.0.0',  # 监听所有网络接口
    'port': 5001,       # 服务端口
    'debug': False,     # 生产环境设为False
    'threaded': True,   # 启用多线程
    'max_content_length': 16 * 1024 * 1024  # 最大请求大小16MB
}

# 加载模型
# MODEL_PATH = "/share/project/lyx/openpi/checkpoints/pi0_agilex_orange/pi0_agilex_orange/50000"
MODEL_PATH = "/share/project/lvhuaihai/openpi/checkpoints/pi0_agilex_orange/pi0_agilex_orange/40000"

# 全局模型变量
policy = None
to_tensor = transforms.ToTensor()

def load_policy():
    """加载openpi模型"""
    global policy
    try:
        logger.info("开始加载openpi模型...")
        # 使用openpi的配置和策略
        config = _config.get_config("pi0_agilex_orange")
        policy = _policy_config.create_trained_policy(config, MODEL_PATH)        
        print(f"policy loaded.")    
        logger.info("openpi模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"openpi模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return False

def decode_image_base64(image_base64):
    """解码base64图片"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = to_tensor(image)
        return image
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise ValueError(f"图片解码失败: {e}")

def process_images(images_dict):
    """处理图片列表，适配openpi的图片格式"""
    try:
        sample_dict = {}
        # 根据openpi的配置调整图片键名
        for k in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
            if k in images_dict:
                sample_dict[k] = decode_image_base64(images_dict[k])
            else:
                logger.warning(f"缺少图片: {k}")

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        raise ValueError(f"处理图片失败: {e}")

    return sample_dict

@app.route('/info', methods=['GET'])
def service_info():
    """服务信息端点"""
    return jsonify({
        "service_name": "OpenPI RoboBrain Robotics API",
        "version": "1.0.0",
        "endpoints": {
            "info": "/info", 
            "infer": "/infer"
        },
        "model_info": {
            "model_path": MODEL_PATH,
            "model_type": "openpi_pi0_agilex_orange"
        },
        "timestamp": time.time()
    })

@app.route('/replay', methods=['GET'])
def replay_api():
    """ground truth qpos API"""
    try: 
        # with h5py.File('/share/project/yunfan/openpi/examples/agilex/replay.h5', 'r') as f:
        #     action = f['actions'][:]
        #     qpos = f['qpos'][:]

        # with open("",'r',encoding='utf-8') as f:
        #     data = json.load(f)
        
        # action = []
        # qpos = data['action']
        # qpos_real = data['action_real']
        # assert np.array(qpos).shape == (8, 50)
        # assert np.array(qpos).shape == (8, 50)

        qpos = np.load('/share/project/chenghy/code/model_eval_real/8.13_pi0/50/pre_action_G50_S4w.npy')
        qpos_real = np.load('/share/project/chenghy/code/model_eval_real/8.13_pi0/50/true_action_G50_S4w.npy')
        
        return jsonify({
            "success": True, 
            # "eepose": action.tolist(),
            "qpos": qpos.tolist(),
            "qpos_real":qpos_real.tolist(),
        })
    except Exception as e:
        logger.error(f"获取ground truth qpos失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/infer', methods=['POST'])
def infer_api():
    """推理API端点"""
    start_time = time.time()
    
    try:
        # 检查模型是否已加载
        if policy is None:
            print("400: 模型未加载，请检查服务状态")
            return jsonify({
                "success": False,
                "error": "模型未加载，请检查服务状态"
            }), 503
        
        # 解析请求数据
        data = request.get_json()
        if not data:
            print("400: 请求数据为空或格式错误!")
            return jsonify({
                "success": False,
                "error": "请求数据为空或格式错误"
            }), 400
        
        if 'eef_pose' not in data:
            print("400: 缺少必需字段: eef_pose!")
            return jsonify({
                "success": False,
                "error": "缺少必需字段: eef_pose"
            }), 400
        
        images = data.get('images')
        state = data.get('state')

        # 处理图片数据
        images_tensor = process_images(images)
        
        # 构建openpi格式的观察数据
        obs = {
            "images": {
                "cam_high": images_tensor['cam_high'],
                "cam_left_wrist": images_tensor['cam_left_wrist'],
                "cam_right_wrist": images_tensor['cam_right_wrist'],
            },
            "state": np.array(state[0]).astype(np.float32),
            "prompt": "task_orange_110_8.11",
        }
        logger.info(f"obs_images: {obs['images']['cam_left_wrist'].shape}")
        logger.info(f"obs_state: {obs['state'].shape}")
        logger.info(f"obs_prompt: {obs['prompt']}")

        # 执行推理
        with torch.no_grad():
            output = policy.infer(obs)
            action = output['actions']

        # 添加处理时间信息
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True, 
            "qpos": action.tolist(), # (50, 14), target joint positions
            "processing_time": processing_time
        }), 200
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"推理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": processing_time
        }), 500

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "success": False,
        "error": "接口不存在",
        "available_endpoints": ["/info", "/infer"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500

if __name__ == '__main__':
    # 加载模型
    if not load_policy():
        logger.error("openpi模型加载失败，服务无法启动")
        sys.exit(1)
    
    print("load policy success")
    # 打印服务信息
    logger.info(f"OpenPI RoboBrain Robotics API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"可用端点:")
    logger.info(f"  - GET  /info    - 服务信息")
    logger.info(f"  - POST /infer   - 推理接口")
    
    # 启动服务
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    ) 

    
```

2.本地控制：

```
**文件名：client_move_from_server_eef.py**      
import base64
import io
import time
import numpy as np
import requests
import cv2
import ipdb
from piper_sdk import C_PiperInterface
from robot_env import RobotEnv
import datetime

CONTROL_MODE = 'joint'  # eepose
def encode_image(img: np.ndarray) -> str:
    """Encode OpenCV image as base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def encode_image_from_frame(frames, cam_name):
    # 获取图像
    img = frames.get(cam_name)

    # 在编码前检查图像是否为空
    if img is not None and img.size > 0:
        # 如果图像有效，则进行编码
        encoded_image = encode_image(img)
        return encoded_image
    else:
        # 如果图像无效，打印一条警告信息或进行其他处理
        print(f"警告：无法获取 {cam_name} 的图像，或者图像为空。")
        return None

# 初始化 RobotEnv（替换成你实际的相机 index 和机械臂 IP）
env = RobotEnv(
    orbbec_serials=[0, 1, 2],
    # realsense_serials=[0],
    arm_ip="can0+can1"
)

joint_command = [ 0.25434  ,  1.8082911, -1.327784 ,  0.7385629,  0.9807441, -0.199704 ,   0.6517 ,
                  -0.20462333,  1.6163499 , -1.0250355 , -0.92851543,  0.7400108 ,  0.37674767, 0.693,
                    ]

print(f"action: {joint_command}")

# time.sleep(2)
# env.control(joint_command)
# time.sleep(1)

try:
    while True:
        # cmd = input("\n按下 'c' 继续一次推理和控制，或按 Ctrl+C 退出：")
        # if cmd.strip().lower() != 'c':
        #     print("[!] 非法输入，输入 'c' 开始下一步。")
        #     continue

        frames, state = env.update_obs_window()
        if state is None or not frames:
            print("[!] 无状态或图像数据，跳过本轮")
            time.sleep(1)
            continue

        qpos = state["qpos"]
        eef_pose = state["eef_pose"]

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        # 保存最新图像到当前目录
        for name, img in frames.items():
            save_path = f"./Log/{name}_{timestamp}.png"
            cv2.imwrite(save_path, img)
            print(f"[Saved] {save_path}")

        encoded_images = {
            "cam_high": encode_image_from_frame(frames, "orbbec_0"),
            "cam_left_wrist": encode_image_from_frame(frames, "orbbec_1"),
            "cam_right_wrist": encode_image_from_frame(frames, "orbbec_2"),
            # "cam_high_realsense": encode_image_from_frame(frames, "realsense_0"),
        }

        data = {
            "state": [qpos.tolist()],           # shape: [1, 14]
            "eef_pose": [eef_pose.tolist()],  # shape: [1, 14]
            "instruction": "pick up the orange",
            "images": encoded_images
        }



        response = requests.get("http://172.16.20.203:5001/replay", json=data, timeout=120)

        # response = requests.post("http://172.16.17.185:8000/infer", json=data, timeout=60)
        print("[√] Response:", response.status_code)

        result = response.json()
        print("[Response JSON]:", result)
        # ipdb.set_trace()
        
        if CONTROL_MODE == 'eepose':
            actions = result.get("eepose", [])
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:20]

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                # ipdb.set_trace()
                action = np.array(act, dtype=np.float32)
                # # 调换前7维（右手）和后7维（左手）
                # if action.shape[0] == 14:
                #     action = np.concatenate([action[7:14], action[0:7]])
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control_eef(action)
                # env.control(action)
                time.sleep(0.1)  # 可根据实际需要调整间隔时间
        elif CONTROL_MODE == 'joint':
            actions = result.get("qpos", [])
            
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:]

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                action = np.array(act, dtype=np.float32)
                ipdb.set_trace()
            
                # # 调换前7维（右手）和后7维（左手）
                # if action.shape[1] == 14:
                # for single_action in action:

                action[:] = np.concatenate([action[:,7:14], action[:, 0:7]], axis=1)
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control(action)
                time.sleep(0.1)  # 可根据实际需要调整间隔时间
except KeyboardInterrupt:
    print("\n[Main] Interrupted by user.")
finally:
    env.shutdown()
    print("[Main] RobotEnv shut down.")

    
```

```
**文件名：robot_env.py**      
import os
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple
from piper_sdk import C_PiperInterface
from pyorbbecsdk import Context, Pipeline, OBSensorType, OBFormat, Config
import pyrealsense2 as rs  

class RealsenseCamera:
    '''
        d405支持分辨率: 
        1280x720: 5,15,30
        848x480: 5,15,30,60,90
        640x360: 5,15,30,60,90
        480x270: 5,15,30,60,90
        424x240: 5,15,30,60,90
        
        d455支持分辨率: 
        1280x720: 5,15,30
        848x480: 5,15,30,60,90
        640x480: 5,15,30,60,90
        640x360: 5,15,30,60,90
        480x270: 5,15,30,60,90
        424x240: 5,15,30,60,90
    '''
    def __init__(self, index: int = 0, width=640, height=480, fps=30):
        DEPTH_RESOLUTION = (width, height)  
        COLOR_RESOLUTION = (width, height)
        DEPTH_FPS = fps
        COLOR_FPS = fps
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0], COLOR_RESOLUTION[1], rs.format.bgr8, COLOR_FPS)
        # config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16, DEPTH_FPS)
    
        # Start streaming
        temp = self.pipeline.start(config)
        # self.align = rs.align(rs.stream.color)
        
        # depth_sensor = temp.get_device().first_depth_sensor()
        # self.depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: " , self.depth_scale)
        
        # frames = self.pipeline.wait_for_frames()
        # aligned_frames = self.align.process(frames)
        # profile = aligned_frames.get_profile()
        # self.intrinsics = rs.video_stream_profile(profile).get_intrinsics()

        for _ in range(3):
            self.get_one_color_frame()

    def get_one_img(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        return color_img, depth_img
    
    # def get_one_color_frame(self):
    #     frames = self.pipeline.wait_for_frames()
    #     aligned_frames = self.align.process(frames)
    #     color_frame = aligned_frames.get_color_frame()
    #     return color_frame

    def get_one_color_frame(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(60)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        data = np.asanyarray(color_frame.get_data()).reshape((color_frame.get_height(), color_frame.get_width(), 3))
        # return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        return data
    
    def get_one_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        return depth_frame
    
    def release(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

class OrbbecCamera:
    def __init__(self, index: int = 0, width=640, height=480, fps=30):
        self.ctx = Context()
        device_list = self.ctx.query_devices()

        if index < 0 or index >= device_list.get_count():
            raise IndexError(f"Orbbec device index {index} out of range. Found {len(device_list)} devices.")

        self.device = device_list.get_device_by_index(index)
        # print(f"[OrbbecCamera] Using device index={index}, serial={self.device.get_serial_number()}")

        self.pipeline = Pipeline(self.device)
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = profile_list.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        config = Config()
        config.enable_stream(self.color_profile)
        self.pipeline.start(config)
        for _ in range(3):
            self.get_one_color_frame()

    def get_one_color_frame(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(60)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        data = np.asanyarray(color_frame.get_data()).reshape((color_frame.get_height(), color_frame.get_width(), 3))
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    def get_one_depth_frame(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None
        data = np.asanyarray(depth_frame.get_data()).reshape((depth_frame.get_height(), depth_frame.get_width(), 1))
        return data

    def stop(self):
        self.pipeline.stop()


class DualArmStateReader:
    def __init__(self, can_left: str, can_right: str):
        self.can_left = can_left
        self.can_right = can_right
        self.piper_left = C_PiperInterface(self.can_left, False)
        self.piper_right = C_PiperInterface(self.can_right, False)
        self.connect()
        # self._initialize_arm(self.piper_left, name="left")
        # self._initialize_arm(self.piper_right, name="right")

    # def _initialize_arm(self, piper: C_PiperInterface, name="arm"):
    #     """切换模式并移动至初始位置"""
    #     initial_joints = [-1.15188663e-02,  5.79031351e-02, -4.69262936e-01,  1.42808183e-01,
    #                      1.13031626e+00, -6.81788497e-02,  6.62983961e-01]
    #     factor = 57324.840764  # 弧度转piper整数
    #     def encode(state):
    #         joints = [round(j * factor) for j in state[:6]]
    #         gripper = round(abs(state[6]) * 1000 * 100)
    #         return joints, gripper

    #     # 编码左右状态
    #     joints_left, gripper_left = encode(initial_joints)
    #     joints_right, gripper_right = encode(initial_joints)

    #     # 单臂控制
    #     piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    #     piper.JointCtrl(*joints_left)
    #     piper.GripperCtrl(gripper_left, 1000, 0x01, 0)
    #     piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)

    #     print(f"[{name}] Arm moved to initial pose.")


    def _enable_arm(self, piper: C_PiperInterface, name="arm"):
        piper.EnableArm(7)
        start_time = time.time()
        timeout = 5
        while time.time() - start_time < timeout:
            status = all([
                piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status,
            ])
            if status:
                print(f"[{name}] Arm enabled.")
                return
            time.sleep(0.5)
        raise TimeoutError(f"[{name}] EnableArm timeout after {timeout}s")

    def _initialize_arm(self, piper: C_PiperInterface, name="arm"):
        initial_joints = [-1.15188663e-02, 5.79031351e-02, -4.69262936e-01, 1.42808183e-01,
                          1.13031626e+00, -6.81788497e-02, 6.62983961e-01]
        factor = 57324.840764

        def encode(state):
            joints = [round(j * factor) for j in state[:6]]
            gripper = round(abs(state[6]) * 1000 * 100)
            return joints, gripper

        joints, gripper = encode(initial_joints)
        piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        piper.JointCtrl(*joints)
        piper.GripperCtrl(gripper, 1000, 0x01, 0)
        # piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        print(f"[{name}] Arm moved to initial pose.")
        time.sleep(0.5)
        
    def _initialize_dual_eef_pose(self, piper_left: C_PiperInterface, piper_right: C_PiperInterface):
        # 初始 EEF Pose（右手前7维，左手后7维）
        # initial_pose = [
        #     -0.8362977849073407, -0.26472032732989265, 0.27504648181181457, -1.0,
        #     0.6405556052423398, -0.9947419557119078, -0.9761430375999883,  # → 右臂
        #     -0.8448706221398707, 0.20283445730307476, 0.2033797131741708,
        #     -0.9870746052288052, 0.6794805403953337, -0.9602500974877654, -0.9776073957552166  # → 左臂
        # ]
        initial_pose = [-0.98268986, -0.02801541, -0.18312784, -0.13389918,  0.7905487 ,
        0.5975816 ,  0.6517    , -0.9385504 , -0.14160725, -0.3147546 ,
        0.09481619,  0.77108294, -0.62963563,  0.693     ]

        def send_pose(piper: C_PiperInterface, pose: list, arm_name: str):
            x, y, z = pose[0:3]
            rx, ry, rz = pose[3:6]
            gripper = pose[6]

            # 转换单位
            x, y, z = int(x * 1e6), int(y * 1e6), int(z * 1e6)
            rx, ry, rz = [int(a * 1000 * 360 / (2 * np.pi)) for a in (rx, ry, rz)]
            gripper = int(abs(gripper * 1000 * 100))

            piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
            piper.EndPoseCtrl(x, y, z, rx, ry, rz)
            piper.GripperCtrl(gripper, 1000, 0x01, 0)
            
            # piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
            print(f"[{arm_name}] EEF moved to initial pose.")

        # 右臂
        send_pose(piper_right, initial_pose[0:7], "right_arm")

        # 左臂
        send_pose(piper_left, initial_pose[7:14], "left_arm")


    def connect(self):
        self.piper_left.ConnectPort()
        self.piper_right.ConnectPort()
        print("[DualArm] Connected to both arms.")
        self._enable_arm(self.piper_left, name="left")
        self._enable_arm(self.piper_right, name="right")
        # self._initialize_dual_eef_pose(self.piper_left, self.piper_right)
        self._initialize_arm(self.piper_left, name="left")
        self._initialize_arm(self.piper_right, name="right")
        time.sleep(0.5)

        # self.connected = True

    def _get_joint_state(self, piper) -> np.ndarray:
        factor = 57324.840764
        joint_msg = piper.GetArmJointMsgs()
        gripper_msg = piper.GetArmGripperMsgs()
        joint_values = [
            joint_msg.joint_state.joint_1.real / factor,
            joint_msg.joint_state.joint_2.real / factor,
            joint_msg.joint_state.joint_3.real / factor,
            joint_msg.joint_state.joint_4.real / factor,
            joint_msg.joint_state.joint_5.real / factor,
            joint_msg.joint_state.joint_6.real / factor,
            gripper_msg.gripper_state.grippers_angle / 1000 / 100,
        ]
        return np.array(joint_values, dtype=np.float32)

    def _get_eef_pose(self, piper) -> np.ndarray:
        """从 Piper 接口读取末端位姿（eef pose）并标准化返回"""
        pose_msg = piper.GetArmEndPoseMsgs()
        gripper_msg = piper.GetArmGripperMsgs()

        # 转换单位并组合：位置（单位 m），姿态（单位 rad），夹爪（归一化）
        eef_pose = [
            pose_msg.end_pose.X_axis * 1e-6,  # mm -> m
            pose_msg.end_pose.Y_axis * 1e-6,
            pose_msg.end_pose.Z_axis * 1e-6,
            pose_msg.end_pose.RX_axis * np.pi / 180,  # deg -> rad
            pose_msg.end_pose.RY_axis * np.pi / 180,
            pose_msg.end_pose.RZ_axis * np.pi / 180,
            gripper_msg.gripper_state.grippers_angle / 1000 / 100  # 保持一致归一化
        ]
        return np.array(eef_pose, dtype=np.float32)

    def get_joint_states(self) -> dict:
        left_state = self._get_joint_state(self.piper_left)
        right_state = self._get_joint_state(self.piper_right)
        return {"left": left_state, "right": right_state}

    def get_eef_poses(self) -> dict:
        def read_pose(piper):
            pose_msg = piper.GetArmEndPoseMsgs()
            gripper_msg = piper.GetArmGripperMsgs()

            pose = [
                pose_msg.end_pose.X_axis * 1e-6,
                pose_msg.end_pose.Y_axis * 1e-6,
                pose_msg.end_pose.Z_axis * 1e-6,
                pose_msg.end_pose.RX_axis * np.pi / (180 * 1000),
                pose_msg.end_pose.RY_axis * np.pi / (180 * 1000),
                pose_msg.end_pose.RZ_axis * np.pi / (180 * 1000),
                gripper_msg.gripper_state.grippers_angle / 1000 / 100,
            ]
            return np.array(pose, dtype=np.float32)

        left_pose = read_pose(self.piper_left)
        right_pose = read_pose(self.piper_right)
        return {"left": left_pose, "right": right_pose}
    
    def send_eef_commands(self, left_pose: np.ndarray, right_pose: np.ndarray, speed: int = 20):
        def execute(piper, pose):
            pose = np.asarray(pose).flatten()
            piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
            piper.EndPoseCtrl(
                int(pose[0] * 1e6),
                int(pose[1] * 1e6),
                int(pose[2] * 1e6),
                int(pose[3] * 180 / np.pi * 1000),
                int(pose[4] * 180 / np.pi * 1000),
                int(pose[5] * 180 / np.pi * 1000),
            )
            piper.GripperCtrl(int(abs(pose[6]) * 1000 * 100), 1000, 0x01, 0)
            # piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
        # import ipdb; ipdb.set_trace()
        execute(self.piper_left, left_pose)
        execute(self.piper_right, right_pose)


    def send_joint_commands(self, left_state: np.ndarray, right_state: np.ndarray, speed: int = 20):
        factor = 57324.840764

        def encode(state):
            state = np.asarray(state).flatten()
            joints = [round(float(j) * factor) for j in state[:6]]
            gripper = round(float(abs(state[6])) * 1000 * 100)
            return joints, gripper

        joints_left, gripper_left = encode(left_state)
        joints_right, gripper_right = encode(right_state)

        self.piper_left.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        self.piper_left.JointCtrl(*joints_left)
        self.piper_left.GripperCtrl(gripper_left, 1000, 0x01, 0)
        time.sleep(0.05)
        # self.piper_left.MotionCtrl_2(0x01, 0x01, speed, 0x00)

        self.piper_right.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        self.piper_right.JointCtrl(*joints_right)
        self.piper_right.GripperCtrl(gripper_right, 1000, 0x01, 0)
        # self.piper_right.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        time.sleep(0.05)


class PiperArm:
    def __init__(self, arm_ip: str = ""):
        self.arm = DualArmStateReader(can_left="can_left", can_right="can_right")

    def get_state(self) -> np.ndarray:
        states = self.arm.get_joint_states()
        return np.concatenate([states["right"], states["left"]], axis=0)

    def get_eef_pose(self) -> np.ndarray:
        """返回左右手末端执行器位姿（例如6DoF + gripper，共7维 * 2）"""
        poses = self.arm.get_eef_poses()  # 应该返回 {"left": np.array(7,), "right": np.array(7,)}
        return np.concatenate([poses["right"], poses["left"]], axis=0)  # shape: (14,)

    def execute(self, action: np.ndarray, wait: bool = True):
        right = action[:7]
        left = action[7:]
        self.arm.send_joint_commands(left, right, speed=20)
    
    def execute_eef(self, action: np.ndarray, wait: bool = True):
        right = action[:7]
        left = action[7:]
        self.arm.send_eef_commands(left, right)
        time.sleep(0.1)
        # self.arm.control_eef_pose(left, right)

class RobotEnv:
    def __init__(
        self,
        realsense_serials: List[int] | None = None,
        orbbec_serials: List[int] | None = None,
        arm_ip: str | None = None,
    ) -> None:
        self._cams: List[Tuple[str, OrbbecCamera]] = []

        for idx in orbbec_serials or []:
            print(f"orbbec_{idx}", OrbbecCamera(index=idx))
            self._cams.append((f"orbbec_{idx}", OrbbecCamera(index=idx)))
        
        for idx in realsense_serials or []:
            print(f"realsense_{idx}", RealsenseCamera(index=idx))
            self._cams.append((f"realsense_{idx}", RealsenseCamera(index=idx)))

        self._arm: PiperArm | None = PiperArm(arm_ip) if arm_ip else None

    def update_obs_window(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        返回：
        - frames: dict，相机名 -> 图像（BGR）
        - state: dict，包含 "qpos" 和 "eef_pose"，若未连接机械臂则为 None
        """
        frames: Dict[str, np.ndarray] = {}
        for name, cam in self._cams:
            img = cam.get_one_color_frame()
            if img is not None:
                frames[name] = img.copy()
                # cv2.imwrite(f"./{name}_latest.jpg", img)

        if self._arm:
            state = {
                "qpos": self._arm.get_state(),        # shape: (14,)
                "eef_pose": self._arm.get_eef_pose()  # shape: (14,)
            }
        else:
            state = None

        return frames, state


    def control(self, action: List[float] | np.ndarray, wait: bool = True):
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")
        self._arm.execute(action, wait=wait)
    
    def control_eef(self, action, wait=True):
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")
        self._arm.execute_eef(action, wait=wait)

    def shutdown(self):
        for _name, cam in self._cams:
            cam.stop()
            
if __name__ == "__main__":
    # 示例主函数入口
    orbbec_serials = [0, 1, 2]  # 替换为你的实际设备序列号
    env = RobotEnv(
        realsense_serials=None,
        orbbec_serials=orbbec_serials,
        arm_ip="can0+can1"
    )

    try:
        while True:
            frames, state = env.update_obs_window()
            for name in frames:
                print(f"[Frame] {name}: shape={frames[name].shape}")
            if state is not None:
                print(f"[Arm State] {state.round(3)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        env.shutdown()
        print("[Main] RobotEnv shut down successfully.")

    
```

3.本地 replay功能：

1. ```
   **文件名：replay_local.py**      
   #!/usr/bin/env python3
   # -*-coding:utf8-*-
   # 注意demo无法直接运行，需要pip安装sdk后才能运行
   from typing import (
       Optional,
   )
   import time
   from piper_sdk import *
   import numpy as np
   import requests
   import h5py
   
   if __name__ == "__main__":
       left_piper = C_PiperInterface("can_left", False)
       right_piper = C_PiperInterface("can_right", False)
       left_piper.ConnectPort()
       right_piper.ConnectPort()
   
       left_piper.GripperCtrl(0,1000,0x01, 0)
       right_piper.GripperCtrl(0,1000,0x01, 0)
   
       
       factor = 1000
       # position = [
       #             57.0, \
       #             0.0, \
       #             215.0, \
       #             0, \
       #             85.0, \
       #             0, \
       #             60]
       position = [
                   0, \
                   0, \
                   0, \
                   0, \
                   0, \
                   0, \
                   0]
       X = round(position[0]*factor)
       Y = round(position[1]*factor)
       Z = round(position[2]*factor)
       RX = round(position[3]*factor)
       RY = round(position[4]*factor)
       RZ = round(position[5]*factor)
       joint_6 = round(position[6]*factor)
   
       left_piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
       left_piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
       left_piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
   
       right_piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
       right_piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
       right_piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
       time.sleep(0.01)
       
       # import ipdb; ipdb.set_trace()
       # response = requests.get("http://172.16.20.117:8000/replay", timeout=60)
       # result = response.json()
       # actions = result.get("action", [])
       with h5py.File("/media/agilex/T7 Shield/Datasets/AGXCompany2/put_pot_lid_on_rack/5730/5730.hdf5", "r") as fin:
           actions = fin["action"][:].tolist()
       actions = np.array(actions)
       print(actions.shape)
       # import ipdb; ipdb.set_trace()
       for idx, action in enumerate(actions):
           left_pose = action[7:]
           right_pose = action[:7]
           # Catch the NaN situation
           try:
               left_result = np.array(
                           [int(left_pose[0] * 1e6),
                           int(left_pose[1] * 1e6),
                           int(left_pose[2] * 1e6),
                           int(left_pose[3] * 180 / np.pi * 1000),
                           int(left_pose[4] * 180 / np.pi * 1000),
                           int(left_pose[5] * 180 / np.pi * 1000),
                           round(abs(left_pose[6]) * 100 * 1000)])
               right_result = np.array(
                           [int(right_pose[0] * 1e3),
                           int(right_pose[1] * 1e3),
                           int(right_pose[2] * 1e3),
                           int(right_pose[3] * 180 / np.pi),
                           int(right_pose[4] * 180 / np.pi),
                           int(right_pose[5] * 180 / np.pi),
                           round(abs(right_pose[6]) * 100)])
   
               right_X = round(right_result[0]*factor)
               right_Y = round(right_result[1]*factor)
               right_Z = round(right_result[2]*factor)
               right_RX = round(right_result[3]*factor)
               right_RY = round(right_result[4]*factor)
               right_RZ = round(right_result[5]*factor)
               right_joint_6 = round(right_result[6]*factor)
   
               # ipdb.set_trace()
               left_piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
               left_piper.EndPoseCtrl(left_result[0],left_result[1],left_result[2],left_result[3],left_result[4],left_result[5])
               left_piper.GripperCtrl(abs(left_result[6]), 1000, 0x01, 0)
   
               right_piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
               right_piper.EndPoseCtrl(right_X,right_Y,right_Z,right_RX,right_RY,right_RZ)
               right_piper.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
               time.sleep(0.1)
           except:
               print(idx, action)
   
   
           
           
           
   
       # while True:
       #     print(piper.GetArmEndPoseMsgs())
       #     # print(piper.GetArmStatus())
       #     import time
       #     count  = count + 1
       #     # print(count)
       #     if(count == 0):
       #         print("1-----------")
       #         position = [
       #             57.0, \
       #             0.0, \
       #             215.0, \
       #             0, \
       #             85.0, \
       #             0, \
       #             0]
       #     # elif(count == 200):
       #     #     print("2-----------")
       #     #     position = [
       #     #         269.0, \
       #     #         0.0, \
       #     #         285.0, \
       #     #         0, \
       #     #         85.0, \
       #     #         0, \
       #     #         0]
       #     elif(count == 400):
       #         print("1-----------")
       #         position = result
       #         count = 0
           
       #     X = round(position[0]*factor)
       #     Y = round(position[1]*factor)
       #     Z = round(position[2]*factor)
       #     RX = round(position[3]*factor)
       #     RY = round(position[4]*factor)
       #     RZ = round(position[5]*factor)
       #     joint_6 = round(position[6]*factor)
       #     print(X,Y,Z,RX,RY,RZ)
       #     # piper.MotionCtrl_1()
       #     piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
       #     piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
       #     piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
       #     time.sleep(0.01)
       #     pass
   
       
   ```

2. 

4.测试脚本：

```
**文件名：test_ctrl_eepose.py**      
#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time
from piper_sdk import *
from robot_env import RobotEnv
import numpy as np
import cv2

if __name__ == "__main__":
    env = RobotEnv(
        orbbec_serials=[0, 1, 2],
        # realsense_serials=[0],
        arm_ip="can0+can1"
    )

    factor = 1000
    position = [
                57.0, \
                0.0, \
                215.0, \
                0, \
                85.0, \
                0, \
                0]

    count = 0
    while True:

        count  = count + 1
        if(count == 0):
            print("1-----------")
            position = [
                57.0, \
                0.0, \
                215.0, \
                0, \
                85.0, \
                0, \
                0]
        elif(count == 200):
            print("2-----------")
            position = [
                57.0, \
                0.0, \
                260.0, \
                0, \
                85.0, \
                0, \
                0]
        elif(count == 400):
            print("1-----------")
            position = [
                57.0, \
                0.0, \
                215.0, \
                0, \
                85.0, \
                0, \
                0]
            count = 0
        
        
        X = round(position[0]*factor)
        Y = round(position[1]*factor)
        Z = round(position[2]*factor)
        RX = round(position[3]*factor)
        RY = round(position[4]*factor)
        RZ = round(position[5]*factor)
        joint_6 = round(position[6]*factor)
        print(X,Y,Z,RX,RY,RZ)

        X, Y, Z = X * 1e-6, Y * 1e-6, Z * 1e-6
        RX, RY, RZ = [a * (2 * np.pi) / (360 * 1000) for a in (RX, RY, RZ)]
        action = [X,Y,Z,RX,RY,RZ,joint_6] * 2

        # action = [-0.98268986, -0.02801541, -0.18312784, -0.13389918,  0.7905487 ,
        # 0.5975816 ,  0.6517    , -0.9385504 , -0.14160725, -0.3147546 ,
        # 0.09481619,  0.77108294, -0.62963563,  0.693     ]
        
        print(f"action: {action}")

        env.control_eef(action)
        
        frames, state = env.update_obs_window()
        if state is None or not frames:
            print("[!] 无状态或图像数据，跳过本轮")
            time.sleep(1)
            continue

        qpos = state["qpos"]
        eef_pose = state["eef_pose"]
        print(f"eef_pose: {eef_pose}")


        time.sleep(0.01)
    

    
```

```
**文件名：test_ctrl_joint.py**      
#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time
from piper_sdk import *
from robot_env import RobotEnv
import numpy as np
import cv2

if __name__ == "__main__":
    env = RobotEnv(
        orbbec_serials=[0, 1, 2],
        # realsense_serials=[0],
        arm_ip="can0+can1"
    )
        
    joint_command = [ -0.20462333,  1.6163499 , -1.0250355 , -0.92851543,  0.7400108 ,  0.37674767, 0.693,
                     0.25434  ,  1.8082911, -1.327784 ,  0.7385629,  0.9807441, -0.199704 ,   0.6517 ]

    print(f"action: {joint_command}")

    time.sleep(1)
    env.control(joint_command)
    time.sleep(1)

    frames, state = env.update_obs_window()
    if state is None or not frames:
        print("[!] 无状态或图像数据，跳过本轮")
        time.sleep(1)

    qpos = state["qpos"]
    eef_pose = state["eef_pose"]
    print(f"eef_pose: {eef_pose}")


    time.sleep(0.01)
    

    
```

```
**文件名：test_orbbec_camera.py**      
# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2

from pyorbbecsdk import Config, Context
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image

ESC_KEY = 27


def main():
    ctx = Context()
    device_list = ctx.query_devices()
    device_list = ctx.query_devices()
    device = device_list.get_device_by_index(int(0))
    config = Config()
    pipeline = Pipeline(device)
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main()

    
```

```
**文件名：test_realsense.py**      

import pyrealsense2 as rs  
import numpy as np
import cv2
class RealsenseCamera:
    '''
        d405支持分辨率: 
        1280x720: 5,15,30
        848x480: 5,15,30,60,90
        640x360: 5,15,30,60,90
        480x270: 5,15,30,60,90
        424x240: 5,15,30,60,90
        
        d455支持分辨率: 
        1280x720: 5,15,30
        848x480: 5,15,30,60,90
        640x480: 5,15,30,60,90
        640x360: 5,15,30,60,90
        480x270: 5,15,30,60,90
        424x240: 5,15,30,60,90
    '''
    def __init__(self):
        DEPTH_RESOLUTION = (640, 480)  
        COLOR_RESOLUTION = (640, 480)
        DEPTH_FPS = 30
        COLOR_FPS = 30
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0], COLOR_RESOLUTION[1], rs.format.bgr8, COLOR_FPS)
        config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16, DEPTH_FPS)
    
        # Start streaming
        temp = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        depth_sensor = temp.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        profile = aligned_frames.get_profile()
        self.intrinsics = rs.video_stream_profile(profile).get_intrinsics()

    def get_one_img(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        return color_img, depth_img
    
    def release(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(depth_frame.get_data())
 
            color_image = np.asanyarray(color_frame.get_data())
 
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            # if key & 0xFF == ord('a'):
            #     save_pointcloud_and_rgb(color_image, depth_image, intrinsics)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

    
```

```
**文件名：utils.py**      
# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
from typing import Union, Any, Optional

import cv2
import numpy as np

from pyorbbecsdk import FormatConvertFilter, VideoFrame
from pyorbbecsdk import OBFormat, OBConvertFormat


def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def determine_convert_format(frame: VideoFrame):
    if frame.get_format() == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    elif frame.get_format() == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    elif frame.get_format() == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    elif frame.get_format() == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    elif frame.get_format() == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    elif frame.get_format() == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    else:
        return None


def frame_to_rgb_frame(frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    print("covert format: {}".format(convert_format))
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print("Convert {} to RGB failed".format(frame.get_format()))
    return rgb_frame


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

    
```

