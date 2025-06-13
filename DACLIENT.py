import grpc
import sys
import os
from server import app_pb2, app_pb2_grpc


def test_image(stub, component_type, image_path):
    """Test single image for defects"""
    requests = [
        app_pb2.ClientRequest(text_input=app_pb2.TextInput(text=component_type)),
        app_pb2.ClientRequest(image_path=app_pb2.ImagePath(path=image_path))
    ]
    
    print(f"Testing image: {image_path}")
    print(f"Component: {component_type}")
    
    label_result = None
    needs_training = False
    received_image = False
    
    try:
        for response in stub.Communicate(iter(requests)):
            if response.HasField('status'):
                label_result = response.status.status
                print(f"Result: {label_result}")
            elif response.HasField('message'):
                message = response.message.message
                print(f"Info: {message}")
                if "not found" in message.lower():
                    needs_training = True
            elif response.HasField('image_data'):
                # Save received image for viewing
                save_received_image(response.image_data, image_path)
                received_image = True
                
    except grpc.RpcError as e:
        print(f"Error: {e}")
    
    return label_result, needs_training, received_image

def save_received_image(image_data, original_path):
    """Save received image to local client folder"""
    try:
        # Use absolute path for received_images folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        received_images_dir = os.path.join(script_dir, "received_images")
        
        # Create folder if it doesn't exist
        if not os.path.exists(received_images_dir):
            os.makedirs(received_images_dir)
        
        # Get original filename and create new name
        original_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(original_name)[0]
        new_filename = os.path.join(received_images_dir, f"{name_without_ext}_received.{image_data.image_format}")
        
        # Save the image
        with open(new_filename, 'wb') as f:
            f.write(image_data.image_bytes)
        
        print(f"Image saved as: {new_filename}")
        
    except Exception as e:
        print(f"Failed to save image: {e}")
        # Don't fail the whole process if image save fails


def train_model(stub, component_type):
    """Train new model for component"""
    requests = [
        app_pb2.ClientRequest(text_input=app_pb2.TextInput(text=component_type)),
        app_pb2.ClientRequest(train_command=app_pb2.TrainCommand(start=True))
    ]
    
    print(f"Training model for: {component_type}")
    
    try:
        for response in stub.Communicate(iter(requests)):
            if response.HasField('message'):
                print(f"Training: {response.message.message}")
    except grpc.RpcError as e:
        print(f"Training error: {e}")



def main(labview_mode=False, image_path=None, component_type=None, auto_train=False):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = app_pb2_grpc.AppServiceStub(channel)

        if not labview_mode:
            # Interactive mode (unchanged)
            print("=== Defect Detection Client ===")
            while True:
                image_path = input("Enter image path (or 'quit'): ").strip().strip('"')
                if image_path.lower() == 'quit':
                    break
                    
                component_type = input("Enter component type: ").strip()
                
                label, needs_training, got_image = test_image(stub, component_type, image_path)
                
                if got_image:
                    print("You can now view the image in the 'received_images' folder")
                
                if needs_training:
                    train_choice = input("Train new model? (y/n): ").strip().lower()
                    if train_choice == 'y':
                        train_model(stub, component_type)
                        print("Try testing the image again now.")
                
                print("-" * 40)
        else:
            # LabVIEW mode - convert output format
            label, needs_training, got_image = test_image(stub, component_type, image_path)
            
            # Convert NG/OK to True/False for LabVIEW compatibility
            if label == "NG":
                print("Defective: True")
            elif label == "OK":
                print("Defective: False")
            
            if needs_training and auto_train:
                train_model(stub, component_type)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        # LabVIEW mode
        image_path = sys.argv[1]
        component_type = sys.argv[2]
        main(labview_mode=True, image_path=image_path, component_type=component_type, auto_train=True)
    else:
        # Interactive mode
        main()