import onnxruntime as ort
import numpy as np
import cv2
import ast # For literal_eval to parse metadata strings
from typing import Union, List, Tuple # For type hinting
import os

def generate_colors(num_classes):
    # Generate distinct colors using HSV color space and convert to BGR
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)  # OpenCV uses hue from 0–179
        hsv_color = np.uint8([[[hue, 255, 255]]])  # Full saturation and value
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr_color))
    return colors
    
class YOLO:
    """
    Performs inference and post-processing for YOLOv8 Segmentation models
    exported to ONNX format using onnxruntime and opencv-python.

    Designed to be called directly with NumPy image arrays:
        model = YOLO('path/to/model.onnx')
        results = model(image_np) # Single image
        results_list = model([image_np1, image_np2]) # Batch of images
    """

    # --- Nested Results Class (No changes needed here) ---
    class Results:
        """
        Holds the results of a segmentation prediction and provides plotting.

        Attributes:
            boxes (list): List of bounding boxes [x1, y1, x2, y2].
            scores (list): List of confidence scores.
            class_ids (list): List of class IDs.
            masks (list): List of binary mask arrays (H, W).
            original_img (np.ndarray): The original image used for prediction (BGR).
            names (dict): Dictionary mapping class IDs to class names.
        """
        def __init__(self, boxes, scores, class_ids, masks, original_img, names):
            self.boxes = boxes
            self.scores = scores
            self.class_ids = class_ids
            self.masks = masks
            self.original_img = original_img # Expecting BGR image
            self.names = names # {0: 'person', 1: 'car', ...}

        def plot(self,
                 show_boxes=True,
                 show_masks=True,
                 show_labels=True,
                 mask_alpha=0.4,
                 # box_color=(0, 255, 0),   # Green (BGR) - Now uses class colors
                 # mask_color=(255, 0, 0),  # Blue (BGR) - Now uses class colors
                 contour_color=(0, 0, 255),# Red (BGR)
                 text_color=(255, 255, 255) # White (BGR)
                ):
            """
            Plots the segmentation results on the original image.

            Args:
                show_boxes (bool): Whether to draw bounding boxes.
                show_masks (bool): Whether to draw masks (color overlay).
                show_labels (bool): Whether to add labels (class name + score).
                mask_alpha (float): Transparency level for mask overlay (0.0 to 1.0).
                contour_color (tuple): Default BGR color for mask contours.
                text_color (tuple): BGR color for labels.

            Returns:
                np.ndarray: The image with plotted results (BGR format).
            """
            if self.original_img is None:
                 print("Warning: Original image is None, cannot plot.")
                 # Return a dummy image or raise error depending on desired behavior
                 return np.zeros((100, 100, 3), dtype=np.uint8) # Example dummy

            vis_img = self.original_img.copy()
            if not self.boxes: # Handle case with no detections
                return vis_img

            # Generate distinct colors for each class if needed, or use a fixed palette
            num_classes = len(self.names)
            # Simple color generation for demonstration using viridis colormap:
            colors = generate_colors(num_classes)  # List of BGR tuples

            # Draw masks first (potentially blended underneath boxes/contours)
            if show_masks and self.masks:
                overlay = vis_img.copy() # Work on a copy for blending
                for i in range(len(self.boxes)):
                    if i >= len(self.masks) or self.masks[i] is None: continue # Skip if mask missing
                    mask = self.masks[i]
                    class_id = self.class_ids[i]
                    current_mask_color = colors[class_id % len(colors)] # Use class-specific color

                    # Apply color to the mask area on the overlay
                    # Ensure mask is boolean or 0/1 for indexing
                    overlay[mask > 0] = current_mask_color

                # Blend the overlay with the original image
                vis_img = cv2.addWeighted(overlay, mask_alpha, vis_img, 1 - mask_alpha, 0)


            # Draw boxes, labels, and contours on top
            for i in range(len(self.boxes)):
                box = self.boxes[i]
                score = self.scores[i]
                class_id = self.class_ids[i]
                mask = self.masks[i] if self.masks and i < len(self.masks) else None
                current_box_color = colors[class_id % len(colors)] # Use class-specific color for box too
                current_contour_color = contour_color # Or make this class-specific too

                x1, y1, x2, y2 = map(int, box)

                # --- Draw Contours ---
                if show_masks and mask is not None:
                    # Ensure mask is uint8 for findContours
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, current_contour_color, 1)

                # --- Draw Bounding Box ---
                if show_boxes:
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), current_box_color, 2)

                # --- Add Label ---
                if show_labels:
                    class_name = self.names.get(class_id, f"Class_{class_id}") # Get name or use ID
                    label = f"{class_name}: {score:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + baseline
                    # Clamp label position to be within image bounds
                    label_y = max(label_height + baseline, label_y)
                    label_x = x1
                    # Draw a filled rectangle behind the text for better visibility
                    cv2.rectangle(vis_img, (label_x, label_y - label_height - baseline), (label_x + label_width, label_y + baseline), current_box_color, cv2.FILLED)
                    cv2.putText(vis_img, label, (label_x, label_y - baseline//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA) # Adjust text position slightly

            return vis_img

        def __len__(self):
            """Return the number of detected objects."""
            return len(self.boxes)

        def __getitem__(self, idx):
            """Allow indexing to get data for a specific detection."""
            if idx >= len(self):
                raise IndexError("Detection index out of range")
            return {
                "box": self.boxes[idx],
                "score": self.scores[idx],
                "class_id": self.class_ids[idx],
                "class_name": self.names.get(self.class_ids[idx], f"Class_{self.class_ids[idx]}"),
                "mask": self.masks[idx] if self.masks else None
            }

    # --- Main Class Methods ---
    def __init__(self, onnx_model_path: str, conf_thres: float = 0.5, iou_thres: float = 0.45, mask_threshold: float = 0.5):
        """
        Initializes the YOLOv8 ONNX Segmentation model handler.

        Args:
            onnx_model_path (str): Path to the ONNX model file.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for Non-Maximum Suppression.
            mask_threshold (float): Threshold for converting probability masks to binary masks.
        """
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.mask_threshold = mask_threshold # Threshold for binary mask
        self.names = {} # Initialize names dictionary
        self.input_width = 640 # Default values, will be updated from model
        self.input_height = 640 # Default values, will be updated from model
        self.num_classes = 0 # Will be updated
        self.mask_coeffs_len = 32 # Default, will be updated

        try:
            # Create ONNX Runtime session
            # Consider adding more provider options like 'TensorrtExecutionProvider' if available and desired
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Filter providers based on availability
            available_providers = ort.get_available_providers()
            valid_providers = [p for p in providers if p in available_providers]
            if not valid_providers:
                raise RuntimeError(f"No valid ONNX Runtime providers found. Available: {available_providers}")

            self.session = ort.InferenceSession(onnx_model_path, providers=valid_providers)
            selected_provider = self.session.get_providers()[0] # Get the provider ORT actually chose
            print(f"ONNX session loaded successfully using provider: {selected_provider}")

            # --- Get metadata ---
            meta = self.session.get_modelmeta().custom_metadata_map
            print("Model Metadata:", meta) # Print fetched metadata for inspection

            # Safely evaluate 'names' metadata
            names_str = meta.get("names")
            if names_str:
                try:
                    self.names = ast.literal_eval(names_str)
                    if not isinstance(self.names, dict):
                        print(f"Warning: Parsed 'names' is not a dict: {self.names}. Resetting to empty.")
                        self.names = {}
                    else:
                        self.names = {int(k): v for k, v in self.names.items()} # Ensure keys are integers
                except Exception as e:
                    print(f"Warning: Could not parse 'names' metadata ({names_str}). Error: {e}")
                    self.names = {}
            else:
                print("Warning: 'names' metadata not found in the ONNX model.")
                self.names = {}

            # Safely evaluate 'imgsz' metadata
            imgsz_str = meta.get("imgsz")
            meta_img_w, meta_img_h = None, None
            if imgsz_str:
                try:
                    imgsz_list = ast.literal_eval(imgsz_str)
                    if isinstance(imgsz_list, (list, tuple)) and len(imgsz_list) == 2:
                        meta_img_h, meta_img_w = int(imgsz_list[0]), int(imgsz_list[1])
                        print(f"Metadata 'imgsz' found: [h={meta_img_h}, w={meta_img_w}]")
                    else:
                        print(f"Warning: Parsed 'imgsz' is not a list/tuple of length 2: {imgsz_list}")
                except Exception as e:
                    print(f"Warning: Could not parse 'imgsz' metadata ({imgsz_str}). Error: {e}")
            else:
                print("Warning: 'imgsz' metadata not found in the ONNX model.")

            # --- Get model inputs (prioritize this for actual dimensions) ---
            model_inputs = self.session.get_inputs()
            if not model_inputs:
                raise ValueError("Failed to get model inputs.")
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape # e.g., [1, 3, 640, 640]

            # Ensure shape has 4 dimensions [batch, channel, height, width]
            if len(self.input_shape) != 4:
                 raise ValueError(f"Unexpected input tensor shape: {self.input_shape}. Expected 4 dimensions.")

            # *** Prioritize input tensor shape for width/height ***
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])
            print(f"Input tensor shape requires: [h={self.input_height}, w={self.input_width}] - This will be used for preprocessing.")

            # Optional: Compare metadata imgsz with input tensor shape
            if meta_img_w is not None and meta_img_h is not None:
                if meta_img_h != self.input_height or meta_img_w != self.input_width:
                    print(f"Warning: Metadata 'imgsz' [h={meta_img_h}, w={meta_img_w}] differs from input tensor shape [h={self.input_height}, w={self.input_width}]. Using tensor shape.")

            # --- Get model outputs ---
            model_outputs = self.session.get_outputs()
            self.output_names = [output.name for output in model_outputs]
            if len(model_outputs) < 2:
                print("Warning: Model has fewer than 2 outputs. Expected format: [predictions, mask_prototypes]. Mask reconstruction might fail.")
                # Attempt to get shape info even if outputs are few
                output0_shape = model_outputs[0].shape # e.g., [1, 116, 8400]
            else:
                output0_shape = model_outputs[0].shape # e.g., [1, 116, 8400] or [1, 37, 8400]
                output1_shape = model_outputs[1].shape # e.g., [1, 32, 160, 160]
                if len(output1_shape) == 4: # Expecting [batch, coeffs, proto_h, proto_w]
                    self.mask_coeffs_len = output1_shape[1] # Should be 32 for standard YOLOv8-Seg
                else:
                    print(f"Warning: Second output shape {output1_shape} is not 4D. Assuming 32 mask coefficients.")
                    self.mask_coeffs_len = 32 # Fallback guess

            # Dynamically determine number of classes
            # Shape is (batch, box[4] + classes[N] + mask_coeffs[M], proposals)
            if len(output0_shape) != 3:
                 raise ValueError(f"Unexpected output tensor 0 shape: {output0_shape}. Expected 3 dimensions.")
            self.num_classes = output0_shape[1] - 4 - self.mask_coeffs_len
            if self.num_classes < 1:
                print(f"Warning: Calculated num_classes ({self.num_classes}) is less than 1 based on output shape {output0_shape} and assumed mask_coeffs={self.mask_coeffs_len}. Check model structure.")
                # Fallback: Use length of names dict if available, otherwise assume 1 class.
                self.num_classes = len(self.names) if self.names else 1 # Use names length or default to 1
                print(f"Adjusted num_classes based on metadata or fallback: {self.num_classes}")


            print(f"Model Input: name='{self.input_name}', shape={self.input_shape}")
            print(f"Model Outputs: names={self.output_names}")
            print(f"Determined {self.num_classes} classes and {self.mask_coeffs_len} mask coefficients.")

            # Validate consistency between num_classes and names dictionary
            if self.names and len(self.names) != self.num_classes:
                print(f"Warning: Number of classes calculated from model output ({self.num_classes}) does not match number of names in metadata ({len(self.names)}). Using calculated value.")
                # Option: Re-create names if mismatch is problematic
                # self.names = {i: f"Class_{i}" for i in range(self.num_classes)}
            elif not self.names and self.num_classes > 0:
                print(f"Warning: Model predicts {self.num_classes} classes, but no class names found in metadata. Using generic Class_ID labels.")
                # Generate generic names if none were found
                self.names = {i: f"Class_{i}" for i in range(self.num_classes)}

        except ort.OrtInvalidArgument as ort_err:
             print(f"ONNX Runtime Error: {ort_err}")
             print("This might be due to model corruption, version mismatch, or incorrect providers.")
             raise RuntimeError(f"ONNX Runtime Error: {ort_err}") from ort_err
        except Exception as e:
            raise RuntimeError(f"Error loading ONNX model or getting metadata: {e}") from e


    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Loads and preprocesses a single image represented as a NumPy array (BGR).

        Args:
            img_bgr (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            Tuple containing:
            - input_tensor (np.ndarray): Preprocessed image tensor for the model.
            - original_img_copy (np.ndarray): A copy of the original input BGR image.
            - preprocess_info (dict): Dictionary with scaling and padding info.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image is empty or invalid.")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
             raise ValueError(f"Input image must be 3-dimensional (H, W, C) with 3 channels (BGR). Got shape {img_bgr.shape}")


        original_img_copy = img_bgr.copy() # Work with a copy
        original_h, original_w = original_img_copy.shape[:2]

        # --- Letterbox resizing and padding ---
        target_h, target_w = self.input_height, self.input_width
        ratio = min(target_w / original_w, target_h / original_h)
        new_w, new_h = int(original_w * ratio), int(original_h * ratio)
        dw, dh = (target_w - new_w) / 2, (target_h - new_h) / 2 # Padding

        # Resize
        if (new_w, new_h) != (original_w, original_h):
            img_resized = cv2.resize(original_img_copy, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = original_img_copy # No resize needed

        # Pad
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) # Gray padding

        # --- Format conversion ---
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

        # Normalize (0-255 -> 0.0-1.0)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Transpose (H, W, C) -> (C, H, W)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension (C, H, W) -> (N, C, H, W)
        input_tensor = np.expand_dims(img_transposed, axis=0)

        # Store preprocessing info for postprocessing adjustments
        preprocess_info = {
            "original_h": original_h,
            "original_w": original_w,
            "ratio": ratio,
            "dw": dw,
            "dh": dh
        }

        return input_tensor, original_img_copy, preprocess_info # Return BGR copy

    def _postprocess(self, outputs: List[np.ndarray], preprocess_info: dict, conf_thres: float, iou_thres: float, mask_thresh: float) -> Tuple[List, List, List, List]:
        """
        Performs post-processing including NMS and mask reconstruction.

        Args:
            outputs (List[np.ndarray]): List of raw outputs from the ONNX model.
            preprocess_info (dict): Dictionary with scaling/padding info from _preprocess.
            conf_thres (float): Confidence threshold for this specific call.
            iou_thres (float): IoU threshold for NMS for this specific call.
            mask_thresh (float): Binary threshold for masks for this specific call.

        Returns:
            Tuple containing:
            - final_boxes (List): List of [x1, y1, x2, y2] boxes.
            - final_scores (List): List of confidence scores.
            - final_class_ids (List): List of class IDs.
            - final_masks (List): List of binary masks (H, W, np.uint8).
        """

        original_h = preprocess_info["original_h"]
        original_w = preprocess_info["original_w"]
        ratio = preprocess_info["ratio"]
        dw = preprocess_info["dw"]
        dh = preprocess_info["dh"]

        # Expecting two outputs: predictions and mask prototypes
        #if len(outputs) < 2:
        #    print("Error: Expected at least 2 outputs from the model for segmentation postprocessing.")
        #    return [], [], [], [] # Return empty lists

        predictions = outputs[0] # Shape (1, 4+N+M, 8400) e.g., (1, 116, 8400)
        # mask_prototypes = outputs[1] # Shape (1, M, proto_h, proto_w) e.g., (1, 32, 160, 160)

        # Check shapes (add batch dimension checks)
        if predictions.shape[0] != 1: 
             print(f"Warning: Expected batch size 1, but got predictions shape {predictions.shape}. Processing first batch element only.")
             # Or raise error if batch > 1 needs special handling

        # Remove batch dim for easier processing
        predictions = predictions[0] # (4+N+M, 8400)
        # mask_prototypes = mask_prototypes[0] # (M, proto_h, proto_w)


        # Check if mask_prototypes shape is valid after removing batch dim
        #if len(mask_prototypes.shape) != 3 or mask_prototypes.shape[0] != self.mask_coeffs_len:
        #     print(f"Warning: Mask prototypes shape {mask_prototypes.shape} unexpected after removing batch dim. Expected ({self.mask_coeffs_len}, H, W). Mask reconstruction might fail.")
        #     mask_prototypes = None # Disable mask processing

        # Transpose predictions: (4+N+M, 8400) -> (8400, 4+N+M)
        detections = np.transpose(predictions) # (8400, 4+N+M)

        boxes = []
        scores = []
        class_ids = []
        mask_coefficient_list = []

        # Precompute mask prototype reshaping if shape is valid
        mask_proto_reshaped = None
        proto_h, proto_w = 0, 0
        mask_prototypes = None
        if mask_prototypes is not None:
            proto_h, proto_w = mask_prototypes.shape[1:] # e.g., 160, 160
            # (M, proto_h*proto_w)
            try:
                 mask_proto_reshaped = mask_prototypes.reshape(self.mask_coeffs_len, -1)
            except ValueError as e:
                 print(f"Error reshaping mask prototypes (shape {mask_prototypes.shape}): {e}. Masks will not be generated.")
                 mask_prototypes = None # Disable masks

        # Loop through 8400 proposals
        for i in range(detections.shape[0]):
            detection = detections[i]

            box_data = detection[0:4] # cx, cy, w, h (relative to padded/resized input)
            class_scores = detection[4 : 4 + self.num_classes]
            mask_coeffs = detection[4 + self.num_classes :]

            # Check consistency
            if len(mask_coeffs) != self.mask_coeffs_len:
                # This indicates a potential mismatch in model output parsing or definition
                # print(f"Warning: Detection {i} has {len(mask_coeffs)} mask coefficients, expected {self.mask_coeffs_len}. Skipping.")
                continue # Skip this detection


            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # Filter based on confidence threshold (using the passed argument)
            if confidence >= conf_thres:
                cx, cy, w, h = box_data
                # Adjust for padding and scaling
                cx = (cx - dw) / ratio
                cy = (cy - dh) / ratio
                w = w / ratio
                h = h / ratio
                # Convert to x1, y1, x2, y2
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                # Clip coordinates to original image dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_w, x2)
                y2 = min(original_h, y2)

                # Ensure box has positive width and height after clipping
                if x2 > x1 and y2 > y1:
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    scores.append(float(confidence))
                    class_ids.append(class_id)
                    if mask_prototypes is not None: # Only store coeffs if protos are valid
                        mask_coefficient_list.append(mask_coeffs)

        # Apply Non-Maximum Suppression
        if not boxes:
            return [], [], [], []

        # Convert boxes to xywh format for NMSBoxes
        boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        # Use passed thresholds for NMS
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)

        # Flatten indices if necessary
        if isinstance(indices, (list, tuple)) and len(indices) > 0 and isinstance(indices[0], (list, tuple)):
             indices = indices.flatten()
        elif not isinstance(indices, np.ndarray):
             indices = np.array(indices).flatten()

        final_boxes = []
        final_scores = []
        final_class_ids = []
        final_masks = []

        # Mask Reconstruction for NMS survivors
        if mask_prototypes is None or mask_proto_reshaped is None:
            print("Skipping mask reconstruction due to invalid/missing prototypes.")
            # Just populate boxes/scores/classes without masks
            for i in indices:
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
                final_masks.append(None) # Append None for mask
        else:
            # Process masks only if prototypes are valid
            for i in indices:
                # Retrieve the filtered detection data
                box = boxes[i] # Original [x1, y1, x2, y2] format
                score = scores[i]
                class_id = class_ids[i]
                mask_coeffs = mask_coefficient_list[i] # Shape (M,)

                # --- Mask Reconstruction ---
                # 1. Matrix Multiply: (M,) @ (M, proto_h*proto_w) -> (proto_h*proto_w,)
                segment = mask_coeffs @ mask_proto_reshaped
                segment = segment.reshape(proto_h, proto_w) # -> (proto_h, proto_w)

                # 2. Sigmoid
                segment = 1 / (1 + np.exp(-segment)) # Probabilities 0-1

                # 3. Resize mask to padded input size
                mask_resized_to_input = cv2.resize(
                    segment,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_LINEAR
                )

                # 4. Remove padding
                unpadded_h = int(self.input_height - 2 * dh)
                unpadded_w = int(self.input_width - 2 * dw)
                # Need careful slicing, ensure indices are valid
                slice_y = slice(int(round(dh)), int(round(dh)) + unpadded_h)
                slice_x = slice(int(round(dw)), int(round(dw)) + unpadded_w)
                mask_unpadded = mask_resized_to_input[slice_y, slice_x]

                # Check if slicing resulted in empty mask (can happen with extreme aspect ratios/padding)
                if mask_unpadded.size == 0:
                     print(f"Warning: Unpadded mask is empty for detection index {i}. Skipping mask.")
                     final_mask_processed = np.zeros((original_h, original_w), dtype=np.uint8) # Empty mask
                else:
                    # 5. Resize mask to original image size
                    final_mask_resized = cv2.resize(
                        mask_unpadded,
                        (original_w, original_h),
                        interpolation=cv2.INTER_LINEAR
                    )

                    # 6. Crop mask using the final bounding box (already in original coords)
                    x1, y1, x2, y2 = box
                    # Ensure box coords are valid int indices
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(original_w, int(x2)), min(original_h, int(y2))

                    if x1 >= x2 or y1 >= y2: # Skip if box is invalid
                         roi_mask_full = np.zeros((0,0), dtype=float) # Empty ROI
                    else:
                         roi_mask_full = final_mask_resized[y1:y2, x1:x2]


                    # 7. Threshold the ROI mask (using passed argument)
                    binary_mask_roi = (roi_mask_full > mask_thresh).astype(np.uint8)

                    # Create a full-size binary mask and place the ROI mask into it
                    full_binary_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                    if x1 < x2 and y1 < y2: # Only place if ROI is valid
                         full_binary_mask[y1:y2, x1:x2] = binary_mask_roi
                    final_mask_processed = full_binary_mask


                # Store final results for this detection
                final_boxes.append(box)
                final_scores.append(score)
                final_class_ids.append(class_id)
                final_masks.append(final_mask_processed) # Store the final binary mask


        return final_boxes, final_scores, final_class_ids, final_masks


    def __call__(self,
                 images: Union[np.ndarray, List[np.ndarray]],
                 conf_thres: float = None,
                 iou_thres: float = None,
                 mask_threshold: float = None
                 ) -> Union[Results, List[Results]]:
        """
        Performs prediction on a single image or a batch of images.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]):
                A single image as a NumPy array (H, W, C - BGR format)
                or a list of images as NumPy arrays.
            conf_thres (float, optional): Overrides the default confidence threshold.
            iou_thres (float, optional): Overrides the default IoU threshold.
            mask_threshold (float, optional): Overrides the default mask threshold.

        Returns:
            Union[YOLO.Results, List[YOLO.Results]]:
                A single Results object if one image was provided,
                or a list of Results objects if a list of images was provided.
                Returns empty Results or list on failure for specific images.
        """
        is_single_image = False
        if isinstance(images, np.ndarray):
            # Basic validation for a single image
            if images.ndim == 3 and images.shape[2] == 3: # HxWxC (BGR)
                images = [images]
                is_single_image = True
            # Handle grayscale (convert to BGR if model expects 3 channels)
            elif images.ndim == 2 and self.input_shape[1] == 3:
                 print("Warning: Received grayscale image, converting to BGR.")
                 images = [cv2.cvtColor(images, cv2.COLOR_GRAY2BGR)]
                 is_single_image = True
            elif images.ndim == 2 and self.input_shape[1] == 1:
                 images = [images] # Keep as is if model expects 1 channel
                 is_single_image = True
            else:
                raise ValueError(f"Input NumPy array has unsupported dimensions or channels. Expected HxWxC (BGR) or HxW (grayscale). Got shape {images.shape}")

        elif isinstance(images, list):
            if not all(isinstance(img, np.ndarray) for img in images):
                raise TypeError("Input list must contain only NumPy arrays.")
            if not images:
                 return [] # Return empty list if input list is empty
            # Convert grayscale images in the list if necessary
            processed_images = []
            for i, img in enumerate(images):
                 if img.ndim == 2 and self.input_shape[1] == 3:
                     print(f"Warning: Image {i} in list is grayscale, converting to BGR.")
                     processed_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                 elif img.ndim == 3 and img.shape[2] == 3:
                     processed_images.append(img) # Assume BGR
                 elif img.ndim == 2 and self.input_shape[1] == 1:
                     processed_images.append(img) # Grayscale ok
                 else:
                     raise ValueError(f"Image {i} in list has unsupported dimensions or channels. Expected HxWxC (BGR) or HxW. Got shape {img.shape}")
            images = processed_images

        else:
            raise TypeError("Input must be a NumPy array (single image) or a list of NumPy arrays.")

        results_list = []
        # Use call-specific thresholds if provided, otherwise use instance defaults
        current_conf = conf_thres if conf_thres is not None else self.conf_threshold
        current_iou = iou_thres if iou_thres is not None else self.iou_threshold
        current_mask_thresh = mask_threshold if mask_threshold is not None else self.mask_threshold

        for original_img_bgr in images:
            original_img_for_results = original_img_bgr # Keep reference in case of error
            try:
                # 1. Preprocess
                input_tensor, original_img_copy, preprocess_info = self._preprocess(original_img_bgr)
                original_img_for_results = original_img_copy # Use the validated copy

                # 2. Run Inference
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

                # 3. Postprocess (pass the potentially overridden thresholds)
                final_boxes, final_scores, final_class_ids, final_masks = self._postprocess(
                    outputs, preprocess_info, current_conf, current_iou, current_mask_thresh
                )

                # 4. Package results
                result = self.Results(
                    final_boxes,
                    final_scores,
                    final_class_ids,
                    final_masks,
                    original_img_copy, # Store the BGR copy used for preprocessing
                    self.names
                )
                results_list.append(result)
                # print(f"Processed image: Detected {len(result)} objects.") # Optional per-image print

            except ValueError as e:
                print(f"Error processing one of the images (ValueError): {e}")
                results_list.append(self.Results([],[],[],[], original_img_for_results, self.names)) # Append empty results, keep original image if possible
            except Exception as e:
                print(f"An unexpected error occurred processing an image: {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging
                results_list.append(self.Results([],[],[],[], original_img_for_results, self.names)) # Append empty results


        return results_list
