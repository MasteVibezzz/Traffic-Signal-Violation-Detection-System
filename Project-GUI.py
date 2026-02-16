from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od  # Import the custom object detection module we documented earlier
import imageio  # For reading and writing video files
import cv2      # OpenCV for image processing

class Window(Frame):
    """
    Main GUI Window for the Traffic Signal Violation Detection System.
    Inherits from tkinter.Frame.
    """
    def __init__(self, master=None):
        """
        Initialize the GUI window, menus, and canvas.
        """
        Frame.__init__(self, master)

        self.master = master
        self.pos = []   # Stores the canvas object IDs for crosshairs (visual feedback)
        self.line = []  # Stores the coordinates (x, y) of the traffic violation line drawn by user
        self.rect = []  # (Unused) Legacy variable for rectangle drawing
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0 # Tracks number of clicks for drawing the line

        # --- Menu Setup ---
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # File Menu
        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file) # Open video file
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        
        # Analyze Menu
        analyze = Menu(menu)
        analyze.add_command(label="Region of Interest", command=self.regionOfInterest) # Step to draw the line
        menu.add_cascade(label="Analyze", menu=analyze)

        # --- Initial Image Setup ---
        # NOTE: Updates hardcoded path to likely location or keep relative if ensuring file exists
        self.filename = "Images/home.jpg" 
        try:
            self.imgSize = Image.open(self.filename)
            self.tkimage =  ImageTk.PhotoImage(self.imgSize)
        except Exception:
             # Fallback if home.jpg is missing or error
            self.tkimage = None

        self.w, self.h = (1366, 768)
        
        # Canvas for displaying images/video preview
        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        if self.tkimage:
            self.canvas.create_image(20, 20, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def open_file(self):
        """
        Handler for File -> Open.
        Opens a file dialog to select a video, grabs the first frame, 
        saves it as a preview, and displays it on the canvas.
        """
        self.filename = filedialog.askopenfilename()

        if not self.filename:
            return

        cap = cv2.VideoCapture(self.filename)

        reader = imageio.get_reader(self.filename)
        fps = reader.get_meta_data()['fps'] 

        ret, image = cap.read() # Read first frame
        
        # NOTE: Hardcoded output path. Ensure this directory exists.
        cv2.imwrite('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/preview.jpg', image)

        self.show_image('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/preview.jpg')


    def show_image(self, frame):
        """
        Displays an image file on the Tkinter Canvas.
        
        Args:
            frame (str): Path to the image file.
        """
        self.imgSize = Image.open(frame)
        self.tkimage =  ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas.destroy() # Re-create canvas to clear previous state

        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def regionOfInterest(self):
        """
        Handler for Analyze -> Region of Interest.
        Enables 'drawing mode' to define the violation line.
        """
        root.config(cursor="plus") # Change cursor to indicate action needed
        self.canvas.bind("<Button-1>", self.imgClick) # Bind left mouse click

    def client_exit(self):
        exit()

    def imgClick(self, event):
        """
        Handles mouse clicks on the canvas.
        Used to draw the violation line by clicking two points.
        """

        if self.counter < 2:
            # Capture click coordinates relative to canvas
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            self.line.append((x, y)) # Add point to line
            
            # Draw visual feedback (red crosshair)
            self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            self.counter += 1

        if self.counter == 2:
            # Two points selected -> Line is defined.
            
            # Unbind mouse events and reset cursor
            self.canvas.unbind("<Button-1>")
            root.config(cursor="arrow")
            self.counter = 0

            # Show created virtual line on the preview image
            print(self.line)
            # print(self.rect) 
            
            img = cv2.imread('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/preview.jpg')
            cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3) # Draw Green Line
            cv2.imwrite('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/copy.jpg', img)
            self.show_image('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/copy.jpg')

            # --- Start Main Processing Loop ---
            self.main_process()
            print("Executed Successfully!!!")

            # Cleanup after processing finishes
            self.line.clear()
            self.rect.clear()
            for i in self.pos:
                self.canvas.delete(i)

    def intersection(self, p, q, r, t):
        """
        Calculates intersection between line segment pq and line segment rt.
        Duplicated logic from object_detection.py for local checks if needed.
        """
        print(p, q, r, t)
        (x1, y1) = p
        (x2, y2) = q

        (x3, y3) = r
        (x4, y4) = t

        a1 = y1-y2
        b1 = x2-x1
        c1 = x1*y2-x2*y1

        a2 = y3-y4
        b2 = x4-x3
        c2 = x3*y4-x4*y3

        if(a1*b2-a2*b1 == 0):
            return False
        print((a1, b1, c1), (a2, b2, c2))
        x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)
        print((x, y))

        if x1 > x2:
            tmp = x1
            x1 = x2
            x2 = tmp
        if y1 > y2:
            tmp = y1
            y1 = y2
            y2 = tmp
        if x3 > x4:
            tmp = x3
            x3 = x4
            x4 = tmp
        if y3 > y4:
            tmp = y3
            y3 = y4
            y4 = tmp

        if x >= x1 and x <= x2 and y >= y1 and y <= y2 and x >= x3 and x <= x4 and y >= y3 and y <= y4:
            return True
        else:
            return False

    def main_process(self):
        """
        Main Loop:
        1. Reads video frame by frame.
        2. Performs Object Detection (YOLOv3).
        3. Checks for Traffic Violations (Intersection with mapped line).
        4. Saves output video.
        """
        video_src = self.filename

        cap = cv2.VideoCapture(video_src)

        reader = imageio.get_reader(video_src)
        fps = reader.get_meta_data()['fps']    
        # Initialize video writer
        writer = imageio.get_writer('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Resources/output/output.mp4', fps = fps)
            
        j = 1 # Frame counter
        while True:
            ret, image = cap.read()
           
            if (type(image) == type(None)):
                writer.close()
                break
            
            image_h, image_w, _ = image.shape
            
            # 1. Preprocess image for YOLO (Resize + Pad)
            new_image = od.preprocess_input(image, od.net_h, od.net_w)

            # 2. Run functionality prediction
            yolos = od.yolov3.predict(new_image)
            boxes = []

            # 3. Decode network output to bounding boxes
            for i in range(len(yolos)):
                boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)

            # 4. Correct box coordinates to original image scale
            od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)

            # 5. Non-Maximum Suppression (Remove duplicates)
            od.do_nms(boxes, od.nms_thresh)     

            # 6. Draw boxes and Check for Violations
            # IMPORTANT: Passes 'self.line' to the object_detection module to check intersections
            image2 = od.draw_boxes(image, boxes, self.line, od.labels, od.obj_thresh, j) 
            
            writer.append_data(image2)

            # Optional: Display output in a window (slower)
            cv2.imshow('Traffic Violation', image2)
            
            print(j)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                writer.close()
                break

            j = j+1

        cv2.destroyAllWindows()

# --- Application Entry Point ---
root = Tk()
app = Window(root)
root.geometry("%dx%d"%(535, 380))
root.title("Traffic Violation")

root.mainloop()