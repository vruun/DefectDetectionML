from concurrent import futures
import grpc
import time
import os
import cv2

import server.app_pb2 as app_pb2
import server.app_pb2_grpc as app_pb2_grpc
import server.model_inference as model_inference
import server.train_model as train_model


class AppService(app_pb2_grpc.AppServiceServicer):
    def __init__(self):
        self.client_states = {}  # Track component_type per client
    
    def Communicate(self, request_iterator, context):
        client_id = id(context)
        self.client_states[client_id] = {'component_type': None}
        
        try:
            for request in request_iterator:
                yield from self._handle_request(request, client_id, context)
        except Exception as e:
            yield app_pb2.ServerResponse(
                message=app_pb2.ServerMessage(message=f"Error: {str(e)}")
            )
        finally:
            if client_id in self.client_states:
                del self.client_states[client_id]
    
    def _handle_request(self, request, client_id, context):
        state = self.client_states[client_id]
        
        if request.HasField('text_input'):
            # Store component type
            state['component_type'] = request.text_input.text.strip()
            yield app_pb2.ServerResponse(
                message=app_pb2.ServerMessage(message=f"Component set to: {state['component_type']}")
            )
            
        elif request.HasField('image_path'):
            # Test image for defects
            if not state['component_type']:
                yield app_pb2.ServerResponse(
                    message=app_pb2.ServerMessage(message="Send component type first")
                )
                return
                
            # Check if component model exists
            exists, _ = self._check_component_exists(state['component_type'])
            if not exists:
                yield app_pb2.ServerResponse(
                    message=app_pb2.ServerMessage(message="Component model not found. Send train command first.")
                )
                return
            
            # Send image to client first
            try:
                image_bytes, image_format = self._load_image_as_bytes(request.image_path.path)
                yield app_pb2.ServerResponse(
                    image_data=app_pb2.ImageData(
                        image_bytes=image_bytes,
                        image_format=image_format
                    )
                )
            except Exception as img_e:
                yield app_pb2.ServerResponse(
                    message=app_pb2.ServerMessage(message=f"Could not load image: {str(img_e)}")
                )
                return
            
            # Run defect detection
            try:
                is_defective, confidence = model_inference.predict_defect(
                    request.image_path.path, state['component_type']
                )
                status = "NG" if is_defective else "OK"
                yield app_pb2.ServerResponse(status=app_pb2.StatusUpdate(status=status))
                
            except Exception as e:
                yield app_pb2.ServerResponse(
                    message=app_pb2.ServerMessage(message=f"Detection failed: {str(e)}")
                )
                
        elif request.HasField('train_command'):
            # Train new model
            if not state['component_type']:
                yield app_pb2.ServerResponse(
                    message=app_pb2.ServerMessage(message="Send component type first")
                )
                return
                
            if request.train_command.start:
                try:
                    success, message = train_model.train_component_model(state['component_type'])
                    if success:
                        yield app_pb2.ServerResponse(
                            message=app_pb2.ServerMessage(message="Model trained successfully")
                        )
                    else:
                        yield app_pb2.ServerResponse(
                            message=app_pb2.ServerMessage(message=f"Training failed: {message}")
                        )
                except Exception as e:
                    yield app_pb2.ServerResponse(
                        message=app_pb2.ServerMessage(message=f"Training error: {str(e)}")
                    )
    
    def _check_component_exists(self, component_type):
        """Check if component model exists"""
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "training_dataset", component_type))
            
            model_path = os.path.join(base_dir, "fine_tuned_model.keras")
            encoder_path = os.path.join(base_dir, "fine_tuned_encoder.keras")
            good_proto_path = os.path.join(base_dir, "good_proto.npy")
            bad_proto_path = os.path.join(base_dir, "bad_proto.npy")

            required_files = [model_path, encoder_path, good_proto_path, bad_proto_path]
            exists = os.path.isdir(base_dir) and all(os.path.exists(f) for f in required_files)
            
            return exists, ""
            
        except Exception as e:
            return False, str(e)
    
    def _load_image_as_bytes(self, image_path):
        """Load image and convert to bytes for transmission"""
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_path}")
            
            # Determine format from file extension
            _, ext = os.path.splitext(image_path.lower())
            if ext in ['.jpg', '.jpeg']:
                image_format = 'jpg'
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
            elif ext == '.png':
                image_format = 'png'
                encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 6]
            else:
                # Default to jpg
                image_format = 'jpg'
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
            
            # Encode image to bytes
            success, encoded_image = cv2.imencode(f'.{image_format}', image, encode_param)
            if not success:
                raise ValueError("Failed to encode image")
            
            image_bytes = encoded_image.tobytes()
            return image_bytes, image_format
            
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    app_pb2_grpc.add_AppServiceServicer_to_server(AppService(), server)
    #server.add_insecure_port('[::]:50051')
    server.add_insecure_port('0.0.0.0:50051')
    server.start()

    print("gRPC server running at port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop(0)


if __name__ == '__main__':
    serve()

