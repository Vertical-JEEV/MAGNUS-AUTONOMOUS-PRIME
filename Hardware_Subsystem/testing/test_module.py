
# importing our class to test and the random module which is need for testing
from comunicate_to_robotic_arm import ArduinoComunicator
import random

def test_module():
    # we get random values between the minimum and maximum values that is possible for each joint
    arduino_comunicator = ArduinoComunicator()
    while True:
        base_angle = random.randint(0, 180)
        shoulder_angle = random.randint(15, 165) 
        elbow_angle = random.randint(0, 180)
        vertical_wrist_angle = random.randint(0, 180)
        rotatory_wrist_angle = 90
        gripper_angle = random.randint(10, 73)

        # using our randomly generated values, we create the string that will be passed to our module
        joint_angle_test_string = f"{base_angle},{shoulder_angle},{elbow_angle},{vertical_wrist_angle},{rotatory_wrist_angle},{gripper_angle}"


        arduino_comunicator.pass_joint_angles_to_arm(joint_angle_test_string)

        # here we check if the robotic arm moved or not, we sort the joint string into the corresponding txt whether they have moved the robotic arm correctly
        print(f"testing string is {joint_angle_test_string}")
        check = input("Did it move correctly ?\n").lower()

        if check == "y":
            with open(r"C:\Users\Sanju\OneDrive\Documents\A_level_subjects\Computer_Science\Project\Project_Devlopment\Hardware_Subsystem\successful_joint_angles.txt", "a") as file:
                file.write(joint_angle_test_string+"\n")
        elif check == "n":
            with open(r"C:\Users\Sanju\OneDrive\Documents\A_level_subjects\Computer_Science\Project\Project_Devlopment\Hardware_Subsystem\failed_joint_angles.txt", "a") as file:
                file.write(joint_angle_test_string+"\n")

        else:
            break

                      
test_module()





