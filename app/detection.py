import cv2
import numpy as np
import pytesseract
from PIL import Image
import math


class ShapeTextDetector:
    def __init__(self):
        self.shapes = []
        self.free_text = []
        self.image_height = 0
        self.image_width = 0
        
    def get_shape_corners(self, shape_type, x, y, w, h):
        """Get corners of a shape based on its type and bounding box"""
        corners = []
        if shape_type == 'rectangle':
            corners = np.array([
                [x, y],           # top-left
                [x + w, y],       # top-right
                [x + w, y + h],   # bottom-right
                [x, y + h]        # bottom-left
            ], dtype=np.int32)
        elif shape_type == 'triangle':
            corners = np.array([
                [x + w//2, y],    # top
                [x, y + h],       # bottom-left
                [x + w, y + h]    # bottom-right
            ], dtype=np.int32)
        elif shape_type == 'circle':
            # For circles, use an octagon approximation
            cx, cy = x + w//2, y + h//2
            r = min(w, h) // 2
            corners = np.array([
                [cx - r, cy],     # left
                [cx - r//2, cy - r//2],  # top-left
                [cx, cy - r],     # top
                [cx + r//2, cy - r//2],  # top-right
                [cx + r, cy],     # right
                [cx + r//2, cy + r//2],  # bottom-right
                [cx, cy + r],     # bottom
                [cx - r//2, cy + r//2]   # bottom-left
            ], dtype=np.int32)
        
        return corners

    def detect_shapes_and_text(self, image_path):
        """Detect shapes and their text content from image"""
        # Read image and get dimensions
        image = cv2.imread(image_path)
        self.image_height, self.image_width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset shapes list
        self.shapes = []
        shape_counts = {'rectangle': 0, 'circle': 0, 'triangle': 0}
        
        # Sort contours by x-coordinate
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Process each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:  # Minimum area threshold
                continue
                
            # Approximate shape
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Get shape type
            vertices = len(approx)
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                shape_type = 'rectangle'
            elif vertices > 4:
                shape_type = 'circle'
            else:
                continue
                
            # Get bounding box and corners
            x, y, w, h = cv2.boundingRect(contour)
            corners = self.get_shape_corners(shape_type, x, y, w, h)
            
            # Extract text from shape region
            roi = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi).strip()
            
            # Create shape ID
            shape_counts[shape_type] += 1
            shape_id = f"{shape_type[0].upper()}{shape_counts[shape_type]}"
            
            # Store shape information
            self.shapes.append({
                'id': shape_id,
                'type': shape_type,
                'text': text,
                'corners': corners,
                'bounds': (x, y, w, h),
                'center': (x + w//2, y + h//2)
            })
            
            # Draw shape outline
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            
            # Draw shape ID with background
            label_bg_pts1 = (x, y-20)
            label_bg_pts2 = (x + 40, y-5)
            cv2.rectangle(image, label_bg_pts1, label_bg_pts2, (255, 255, 255), -1)
            cv2.putText(image, shape_id, (x+5, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw text inside shape
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (w - text_size[0])//2
            text_y = y + (h + text_size[1])//2
            cv2.putText(image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Detect free text
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for shape in self.shapes:
            x, y, w, h = shape['bounds']
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
        mask = cv2.bitwise_not(mask)
        free_text_img = cv2.bitwise_and(gray, gray, mask=mask)
        self.free_text = pytesseract.image_to_string(free_text_img).strip()
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        return pil_image, self.shapes, self.free_text

    def check_box_intersection(self, box1, box2):
        """Check if two bounding boxes intersect"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or
                   x2 + w2 < x1 or
                   y1 + h1 < y2 or
                   y2 + h2 < y1)

    def line_intersection(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def find_non_overlapping_position(self, base_x, base_y, width, height, shape_type):
            """Find the nearest non-overlapping position considering all shapes"""
            # Define spiral search pattern
            def get_spiral_points(max_radius, step=20):
                points = []
                t = 0
                while t < max_radius:
                    t += step / (2 * math.pi)
                    x = t * math.cos(t)
                    y = t * math.sin(t)
                    points.append((int(x), int(y)))
                return points

            # Try different positions in a spiral pattern
            max_radius = min(self.image_width, self.image_height) / 2
            spiral_points = get_spiral_points(max_radius)
            padding = 20

            for dx, dy in spiral_points:
                new_x = base_x + dx
                new_y = base_y + dy

                # Check image boundaries
                if (new_x - width/2 < padding or 
                    new_x + width/2 > self.image_width - padding or
                    new_y - height/2 < padding or 
                    new_y + height/2 > self.image_height - padding):
                    continue

                # Get corners for the proposed position
                x = int(new_x - width/2)
                y = int(new_y - height/2)
                corners = self.get_shape_corners(shape_type, x, y, int(width), int(height))

                # Check for collisions with ALL shapes
                if not self.check_collision(corners):
                    return new_x, new_y

            # If no position found, try with reduced size
            if width > 100 and height > 50:
                new_width = width * 0.8
                new_height = height * 0.8
                return self.find_non_overlapping_position(base_x, base_y, new_width, new_height, shape_type)

            raise ValueError("Could not find suitable position for new shape")


    def check_collision(self, new_corners):
        """Check if new shape collides with existing shapes with increased margin"""
        # Add safety margin around shapes
        margin = 20
        
        # Convert new_corners to the correct format for OpenCV
        new_corners = np.array(new_corners, dtype=np.int32)
        
        # Get bounding box of new shape
        x, y, w, h = cv2.boundingRect(new_corners)
        # Expand bounding box by margin
        x -= margin
        y -= margin
        w += 2 * margin
        h += 2 * margin
        
        for shape in self.shapes:
            existing_corners = np.array(shape['corners'], dtype=np.int32)
            ex, ey, ew, eh = cv2.boundingRect(existing_corners)
            
            # Check bounding box intersection first (faster)
            if (x < ex + ew and x + w > ex and
                y < ey + eh and y + h > ey):
                return True
            
        return False

    def add_shape(self, image_path, shape_type, text, ref_shape1_id, ref_shape2_id=None, position='between'):
            """Add new shape to image with consistent styling"""
            # Find reference shapes
            ref_shape1 = next((s for s in self.shapes if s['id'] == ref_shape1_id), None)
            if not ref_shape1:
                raise ValueError(f"Reference shape {ref_shape1_id} not found")
                
            ref_shape2 = None
            if ref_shape2_id:
                ref_shape2 = next((s for s in self.shapes if s['id'] == ref_shape2_id), None)
                if not ref_shape2:
                    raise ValueError(f"Reference shape {ref_shape2_id} not found")

            # Calculate initial position and size
            initial_x, initial_y, width, height = self.calculate_available_space(
                ref_shape1, ref_shape2, position
            )
            
            # Try to find non-overlapping position
            try:
                center_x, center_y = self.find_non_overlapping_position(
                    initial_x, initial_y, width, height, shape_type
                )
            except ValueError as e:
                raise ValueError("Could not find suitable position for new shape")

            # Calculate shape coordinates
            x = int(center_x - width/2)
            y = int(center_y - height/2)
            
            # Get corners for collision detection
            corners = self.get_shape_corners(shape_type, x, y, int(width), int(height))
            
            # Check for collisions
            if self.check_collision(corners):
                raise ValueError("Cannot place shape without overlap")

            # Draw new shape
            image = cv2.imread(image_path)
            line_thickness = 2  # Consistent line thickness
            
            if shape_type == 'rectangle':
                cv2.rectangle(image, 
                            (x, y), 
                            (x + int(width), y + int(height)), 
                            (0, 255, 0), 
                            line_thickness)
            elif shape_type == 'circle':
                center = (int(center_x), int(center_y))
                radius = int(min(width, height)/2)
                cv2.circle(image, center, radius, (0, 255, 0), line_thickness)
            elif shape_type == 'triangle':
                points = np.array([
                    [x + width/2, y],
                    [x, y + height],
                    [x + width, y + height]
                ], np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], True, (0, 255, 0), line_thickness)

            # Generate new shape ID
            shape_count = len([s for s in self.shapes if s['type'] == shape_type]) + 1
            new_shape_id = f"{shape_type[0].upper()}{shape_count}"
            
            # Draw shape ID with background
            text_size = cv2.getTextSize(new_shape_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_width = text_size[0] + 10
            label_height = text_size[1] + 10
            
            # Position label background
            label_x = x + (int(width) - label_width) // 2
            label_y = y - label_height - 5
            
            # Draw white background for ID
            cv2.rectangle(image, 
                        (label_x, label_y), 
                        (label_x + label_width, label_y + label_height), 
                        (255, 255, 255), 
                        -1)
            
            # Draw ID text centered in background
            cv2.putText(image, 
                    new_shape_id, 
                    (label_x + 5, label_y + label_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    1)

            # Draw text inside shape with background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (int(width) - text_size[0])//2
            text_y = y + (int(height) + text_size[1])//2
            
            # Add white background for shape text
            text_bg_padding = 4
            cv2.rectangle(image,
                        (text_x - text_bg_padding, text_y - text_size[1] - text_bg_padding),
                        (text_x + text_size[0] + text_bg_padding, text_y + text_bg_padding),
                        (255, 255, 255),
                        -1)
            
            # Draw shape text
            cv2.putText(image, 
                    text, 
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1)

            # Add to shapes list
            self.shapes.append({
                'id': new_shape_id,
                'type': shape_type,
                'text': text,
                'corners': corners,
                'bounds': (x, y, int(width), int(height)),
                'center': (int(center_x), int(center_y))
            })

            # Convert to PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            return pil_image, new_shape_id

    def calculate_available_space(self, ref_shape1, ref_shape2=None, position='between'):
            """Calculate available space that doesn't overlap with ANY existing shapes"""
            # Get reference shape dimensions
            x1, y1, w1, h1 = ref_shape1['bounds']
            center_1_x = x1 + w1//2
            center_1_y = y1 + h1//2
            margin = 30  # Increased margin for better spacing

            # Initialize space variables
            center_x = 0
            center_y = 0
            available_width = 0
            available_height = 0

            if ref_shape2 and position == 'between':
                x2, y2, w2, h2 = ref_shape2['bounds']
                center_2_x = x2 + w2//2
                center_2_y = y2 + h2//2

                # Calculate initial center point between shapes
                center_x = (center_1_x + center_2_x) // 2
                center_y = (center_1_y + center_2_y) // 2

                # Calculate distance between shapes
                dx = abs(center_2_x - center_1_x)
                dy = abs(center_2_y - center_1_y)
                
                # Find all shapes that intersect with the bounding box of the two reference shapes
                min_x = min(x1, x2)
                max_x = max(x1 + w1, x2 + w2)
                min_y = min(y1, y2)
                max_y = max(y1 + h1, y2 + h2)
                
                # Initial available space is the distance between shapes
                available_width = dx * 0.5  # Use 50% of distance between shapes
                available_height = min(h1, h2) * 0.5  # Use 50% of the smaller height
                
                # Check for other shapes in this area
                interfering_shapes = [
                    s for s in self.shapes 
                    if s != ref_shape1 and s != ref_shape2 and
                    self.check_box_intersection(
                        (min_x, min_y, max_x - min_x, max_y - min_y),
                        s['bounds']
                    )
                ]
                
                # Reduce available space if there are interfering shapes
                if interfering_shapes:
                    available_width *= 0.8  # Further reduce space if there are other shapes
                    available_height *= 0.8

            else:
                # Position relative to single shape
                box_margin = margin * 2  # Double margin for single reference positioning
                
                if position == 'above':
                    center_x = center_1_x
                    center_y = y1 - margin - h1//2
                    available_width = w1 * 0.7
                    available_height = min(y1 - margin, h1 * 0.7)
                elif position == 'below':
                    center_x = center_1_x
                    center_y = y1 + h1 + margin + h1//2
                    available_width = w1 * 0.7
                    available_height = min(self.image_height - (y1 + h1 + margin), h1 * 0.7)
                elif position == 'left':
                    center_x = x1 - margin - w1//2
                    center_y = center_1_y
                    available_width = min(x1 - margin, w1 * 0.7)
                    available_height = h1 * 0.7
                elif position == 'right':
                    center_x = x1 + w1 + margin + w1//2
                    center_y = center_1_y
                    available_width = min(self.image_width - (x1 + w1 + margin), w1 * 0.7)
                    available_height = h1 * 0.7
                
                # Check for interfering shapes in the target area
                search_box = (
                    center_x - available_width//2 - box_margin,
                    center_y - available_height//2 - box_margin,
                    available_width + 2*box_margin,
                    available_height + 2*box_margin
                )
                
                interfering_shapes = [
                    s for s in self.shapes 
                    if s != ref_shape1 and 
                    self.check_box_intersection(search_box, s['bounds'])
                ]
                
                # Reduce space if there are interfering shapes
                if interfering_shapes:
                    available_width *= 0.8
                    available_height *= 0.8

            # Ensure minimum and maximum sizes
            min_width = 100
            min_height = 50
            max_width = self.image_width // 4
            max_height = self.image_height // 4

            available_width = max(min_width, min(available_width, max_width))
            available_height = max(min_height, min(available_height, max_height))

            # Adjust center position to ensure shape stays within image bounds
            center_x = max(available_width//2 + margin, min(self.image_width - available_width//2 - margin, center_x))
            center_y = max(available_height//2 + margin, min(self.image_height - available_height//2 - margin, center_y))

            return center_x, center_y, available_width, available_height

