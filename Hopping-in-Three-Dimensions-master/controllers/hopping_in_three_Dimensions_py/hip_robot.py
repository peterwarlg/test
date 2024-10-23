import math
import copy
import numpy
import numpy as np
from controller import Robot, Motor, PositionSensor, TouchSensor, InertialUnit
from scipy.spatial.transform import Rotation as R

LOADING = 0x00  # 落地
COMPRESSION = 0x01  # 压缩腿
THRUST = 0x02  # 伸长腿
UNLOADING = 0x03  # 离地
FLIGHT = 0x04  # 飞行


class HipRobot:
    def __init__(self):
        # Robot class的instance
        self.robot = Robot()

        # deivces init
        self.spring_motor: Motor \
            = self.robot.getMotor("linear motor")
        self.spring_pos_sensor: PositionSensor \
            = self.robot.getPositionSensor("position sensor")
        self.touch_sensor: TouchSensor \
            = self.robot.getTouchSensor("touch sensor")
        self.IMU: InertialUnit \
            = self.robot.getInertialUnit("inertial unit")
        self.X_motor: Motor \
            = self.robot.getMotor("X rotational motor")
        self.X_motor_position_sensor: PositionSensor \
            = self.robot.getPositionSensor("X position sensor")
        self.Z_motor: Motor \
            = self.robot.getMotor("Z rotational motor")
        self.Z_motor_position_sensor: PositionSensor \
            = self.robot.getPositionSensor("Z position sensor")

        self.time_step = 2
        self.spring_pos_sensor.enable(self.time_step)
        self.X_motor_position_sensor.enable(self.time_step)
        self.Z_motor_position_sensor.enable(self.time_step)
        self.IMU.enable(self.time_step)
        self.touch_sensor.enable(self.time_step)

        # 机器人属性
        self.spring_normal_length: float = 1.2  # *弹簧原长
        self.k_spring: float = 2000.0  # 弹簧刚度
        self.F_thrust: float = 100.0  # THRUST推力
        self.k_xz_dot: float = 0.072  # 净加速度系数
        self.r_threshold: float = 0.95  # 状态机在脱离LOADING和进入UNLOADING状态时，腿长阈值判断
        self.v: float = 0.2  # 机器人水平运动速度
        self.k_pose_p: float = 0.8  # 姿态控制时的kp
        self.k_pose_v: float = 0.025  # 姿态控制时的kv
        self.k_leg_p: float = 6.0  # 腿部控制时的kp
        self.k_leg_v: float = 0.8  # 腿部控制时的kv

        # 机器人状态
        # euler angle
        self.eulerAngle: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.eulerAngle_dot: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.B_under_H: np.ndarray = np.eye(3, dtype=float)  # body->world
        self.H_under_B: np.ndarray = np.eye(3, dtype=float)  # world->body

        # r x z
        self.jointPoint: np.ndarray = np.array([0.8, 0.0, 0.0])
        self.jointPoint_dot: np.ndarray = np.array([0.0, 0.0, 0.0])

        # x y z
        self.workPoint_B: np.ndarray = np.array(np.zeros(3, dtype=float))
        self.workPoint_H: np.ndarray = np.array(np.zeros(3, dtype=float))

        self.workPoint_B_desire: np.ndarray = np.array(np.zeros(3, dtype=float))
        self.workPoint_H_desire: np.ndarray = np.array(np.zeros(3, dtype=float))

        self.is_foot_touching_flg: bool = True
        self.Ts: float = 0.0
        self.x_dot: float = 0.0
        self.z_dot: float = 0.0
        self.x_dot_desire: float = 0.0
        self.z_dot_desire: float = 0.0

        self.system_ms: int = 0
        self.stateMachine: int = THRUST

        self.pre_is_foot_touching_flg = False
        self.stance_start_ms: int = 0
        self.debug = []
        self.pre_x_dot = 0.0
        self.pre_z_dot = 0.0

    # 函数功能：设置虚拟弹簧的力
    # 注意：弹簧力正方向定义为：腿不动，将机身向y轴方向推动的力
    def set_spring_force(self, force: float):
        self.spring_motor.setForce(-force)

    def set_X_torque(self, torque: float) -> None:
        self.X_motor.setTorque(torque)

    def set_Z_torque(self, torque: float) -> None:
        self.Z_motor.setTorque(torque)

    def get_spring_length(self) -> float:
        length = self.spring_pos_sensor.getValue()
        return -length + 0.8

    def get_X_motor_angle(self) -> float:
        angle = self.X_motor_position_sensor.getValue()
        return angle * 180.0 / math.pi

    def get_Z_motor_angle(self) -> float:
        angle = self.Z_motor_position_sensor.getValue()
        return angle * 180.0 / math.pi

    def is_foot_touching(self) -> bool:
        return self.touch_sensor.getValue()

    def get_IMU_Angle(self) -> np.ndarray:
        rpy = self.IMU.getRollPitchYaw()
        return np.array([rpy[0] * 180.0 / math.pi, rpy[1] * 180.0 / math.pi, rpy[2] * 180.0 / math.pi])

    def robot_control(self):
        dx: float = self.spring_normal_length - self.jointPoint[0]
        F_spring: float = dx * self.k_spring
        if self.stateMachine == THRUST:
            F_spring = F_spring + self.F_thrust
        self.set_spring_force(F_spring)

        # 控制臀部扭矩力
        if self.stateMachine == LOADING or self.stateMachine == UNLOADING:
            self.set_X_torque(0.0)
            self.set_Z_torque(0.0)

        if self.stateMachine == COMPRESSION or self.stateMachine == THRUST:
            tx = -(-self.k_pose_p * self.eulerAngle[0] - self.k_pose_v * self.eulerAngle_dot[0])
            tz = -(-self.k_pose_p * self.eulerAngle[1] - self.k_pose_v * self.eulerAngle_dot[1])
            self.set_X_torque(tx)
            self.set_Z_torque(tz)

        # FLIGHT的时候，控制足底移动到落足点
        if self.stateMachine == FLIGHT:
            r = self.jointPoint[0]
            x_f = self.x_dot * self.Ts / 2.0 + self.k_xz_dot * (self.x_dot - self.x_dot_desire)
            z_f = self.z_dot * self.Ts / 2.0 + self.k_xz_dot * (self.z_dot - self.z_dot_desire)

            y_f = -math.sqrt(r * r - x_f * x_f - z_f * z_f)
            self.workPoint_H_desire = np.array([x_f, y_f, z_f])
            self.workPoint_B_desire = self.H_under_B @ self.workPoint_H_desire
            x_f_B = self.workPoint_H_desire.data[0]
            y_f_B = self.workPoint_H_desire.data[1]
            z_f_B = self.workPoint_H_desire.data[2]
            x_angle_desire = math.atan(z_f_B / y_f_B) * 180.0 / math.pi
            z_angle_desire = math.asin(x_f_B / r) * 180.0 / math.pi

            x_angle = self.jointPoint[1]
            z_angle = self.jointPoint[2]
            x_angle_dot = self.jointPoint_dot[1]
            z_angle_dot = self.jointPoint_dot[2]
            tx = -self.k_leg_p * (x_angle - x_angle_desire) - self.k_leg_v * x_angle_dot
            tz = -self.k_leg_p * (z_angle - z_angle_desire) - self.k_leg_v * z_angle_dot

            self.set_X_torque(tx)
            self.set_Z_torque(tz)
        pass

    def update_robot_state(self):
        self.system_ms = self.system_ms + self.time_step

        self.is_foot_touching_flg = self.is_foot_touching()

        now_IMU = self.get_IMU_Angle()
        for i in range(3):
            self.eulerAngle_dot[i] = \
                (now_IMU[i] - self.eulerAngle[i]) / (0.001 * self.time_step)
            self.eulerAngle[i] = now_IMU[i]

        r = R.from_euler('xzy', now_IMU, degrees=False)
        self.B_under_H = np.array(r.as_matrix())
        self.H_under_B = np.transpose(self.B_under_H)
        print('IMU: \n', now_IMU)
        print('matrix: \n', r.as_matrix())
        now_r = self.get_spring_length()
        now_r_dot = (now_r - self.jointPoint[0]) / (0.001 * self.time_step)
        self.jointPoint[0] = now_r
        self.jointPoint_dot[0] = self.jointPoint_dot[0] * 0.5 + now_r_dot * 0.5

        now_x_motor = self.get_X_motor_angle()
        now_x_motor_dot = self.get_X_motor_angle() - self.jointPoint[1]
        self.jointPoint[1] = now_x_motor
        self.jointPoint_dot[1] = now_x_motor_dot * 0.5 + self.jointPoint_dot[1] * 0.5

        now_z_motor = self.get_Z_motor_angle()
        now_z_motor_dot = self.get_Z_motor_angle() - self.jointPoint[2]
        self.jointPoint[2] = now_z_motor
        self.jointPoint_dot[2] = now_z_motor_dot * 0.5 + self.jointPoint_dot[2] * 0.5

        self.update_xz_dot()
        self.update_last_Ts()
        self.updateRobotStateMachine()

    def forward_kinect(self):
        # forward
        tx = self.jointPoint[1] * math.pi / 180.0
        tz = self.jointPoint[2] * math.pi / 180.0
        r = self.jointPoint[0]

        self.workPoint_B[0] = r * math.sin(tz)
        self.workPoint_B[1] = -r * math.cos(tz) * math.cos(tx)
        self.workPoint_B[2] = -r * math.cos(tz) * math.sin(tx)

    def update_xz_dot(self):
        self.forward_kinect()

        pre_x, pre_z = self.workPoint_H[0], self.workPoint_H[2]
        self.workPoint_H = self.B_under_H @ self.workPoint_B
        now_x, now_z = self.workPoint_H[0], self.workPoint_H[2]

        now_x_dot = - (now_x - pre_x) / (0.001 * self.time_step)
        now_z_dot = - (now_z - pre_z) / (0.001 * self.time_step)

        now_x_dot = self.pre_x_dot * 0.5 + now_x_dot * 0.5
        now_z_dot = self.pre_z_dot * 0.5 + now_z_dot * 0.5
        self.pre_x_dot = now_x_dot
        self.pre_z_dot = now_z_dot

        if self.stateMachine == COMPRESSION or self.stateMachine == THRUST:
            self.x_dot = now_x_dot
            self.z_dot = now_z_dot

    def update_last_Ts(self):
        if not self.pre_is_foot_touching_flg and self.is_foot_touching_flg:
            self.stance_start_ms = self.system_ms
        if self.pre_is_foot_touching_flg and not self.is_foot_touching_flg:
            stance_end_time = self.system_ms
            self.Ts = 0.001 * (stance_end_time - self.stance_start_ms)
        self.pre_is_foot_touching_flg = self.is_foot_touching_flg

    def updateRobotStateMachine(self):
        if self.stateMachine == LOADING:
            if self.jointPoint[0] < self.spring_normal_length * self.r_threshold:
                self.stateMachine = COMPRESSION
            return
        elif self.stateMachine == COMPRESSION:
            if self.jointPoint_dot[0] > 0:
                self.stateMachine = THRUST
            return

        elif self.stateMachine == THRUST:
            if self.jointPoint[0] > self.spring_normal_length * self.r_threshold:
                self.stateMachine = UNLOADING
            return
        elif self.stateMachine == UNLOADING:
            if not self.is_foot_touching_flg:
                self.stateMachine = FLIGHT
            return
        elif self.stateMachine == FLIGHT:
            if self.is_foot_touching_flg:
                self.stateMachine = LOADING
            return
        else:
            return
        pass
