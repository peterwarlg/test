"""hopping_in_three_Dimensions_py controller."""

# You may need to import some classes of the controller module. Ex:

from hip_robot import HipRobot

angle = 0.0

if __name__ == '__main__':
    # test
    hip_robot = HipRobot()

    while hip_robot.robot.step(hip_robot.time_step) != -1:
        # hip_robot.X_motor.setPosition(angle)
        # val = hip_robot.X_motor_position_sensor.getValue()
        # angle = angle + 0.001
        #
        # hip_robot.Z_motor.setPosition(0.5)
        hip_robot.update_robot_state()
        hip_robot.robot_control()
        # print(hip_robot.x_dot, hip_robot.z_dot)

        pass

# Enter here exit cleanup code.
