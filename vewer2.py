import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor


class DualRGBImageViewer:
    def __init__(self, image_path1, image_path2):
        # Load image 1 (OpenCV uses BGR, so convert to RGB)
        self.image1 = cv2.imread(image_path1)
        if self.image1 is None:
            raise ValueError(f"Could not load image file 1: {image_path1}")
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)

        # Load image 2 (OpenCV uses BGR, so convert to RGB)
        self.image2 = cv2.imread(image_path2)
        if self.image2 is None:
            raise ValueError(f"Could not load image file 2: {image_path2}")
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)

        # Get height and width of each image
        self.height1, self.width1 = self.image1.shape[:2]
        self.height2, self.width2 = self.image2.shape[:2]

        # Create two subplots with matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Display image 1 on the left
        self.im1 = self.ax1.imshow(self.image1)
        self.ax1.set_title("Image 1 - Hover to display RGB values")

        # Display image 2 on the right
        self.im2 = self.ax2.imshow(self.image2)
        self.ax2.set_title("Image 2 - Hover to display RGB values")

        # Text for displaying RGB values (for each image)
        self.text1 = self.ax1.text(
            0.02,
            0.98,
            "",
            transform=self.ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.text2 = self.ax2.text(
            0.02,
            0.98,
            "",
            transform=self.ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set up mouse event
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Display cursors (for each image)
        self.cursor1 = Cursor(self.ax1, useblit=True, color="red", linewidth=1)
        self.cursor2 = Cursor(self.ax2, useblit=True, color="red", linewidth=1)

    def on_mouse_move(self, event):
        """Event handler for mouse movement"""
        # Mouse movement within image 1 area
        if event.inaxes == self.ax1:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.width1 and 0 <= y < self.height1:
                r, g, b = self.image1[y, x]
                rgb_text = f"Coordinates: ({x}, {y})\nRGB: ({r}, {g}, {b})\nHex: #{r:02x}{g:02x}{b:02x}"
                self.text1.set_text(rgb_text)
                self.fig.canvas.draw_idle()

        # Mouse movement within image 2 area
        elif event.inaxes == self.ax2:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.width2 and 0 <= y < self.height2:
                r, g, b = self.image2[y, x]
                rgb_text = f"Coordinates: ({x}, {y})\nRGB: ({r}, {g}, {b})\nHex: #{r:02x}{g:02x}{b:02x}"
                self.text2.set_text(rgb_text)
                self.fig.canvas.draw_idle()

    def show(self):
        """Display the images"""
        plt.tight_layout()
        plt.show()


def main():
    print("Display two images side by side")

    # # Specify the path for the first image file
    # image_path1 = input("Enter the path for the first image file: ")

    # # Specify the path for the second image file
    # image_path2 = input("Enter the path for the second image file: ")

    base_path = "../Data/ai-gen-image-stats/"
    fp1 = "original/ctcs-resized-bilinear-512/160606-L-4-2-tif_img-no150_y525_x38.png"
    fp2 = "lora-trained-linear/final-model-degamma/seed8_rsd_lora.png"
    # fp = "lora-trained-linear-resized/final-model_degamma/seed2_rsd_lora.png"
    # fp = "lora-trained-gamma/ctcs-lora-degamma/seed16_rsd_lora.png"
    base_path = "./out/"
    fp1 = "ctcs-lora-linear/seed8_rsd_lora.png"
    fp2 = "ctcs-lora-linear-color-adjusted/seed8_rsd_lora.png"
    image_path1 = base_path + fp1
    image_path2 = base_path + fp2

    try:
        # Create an instance of DualRGBImageViewer
        viewer = DualRGBImageViewer(image_path1, image_path2)

        # Display the images
        print("Displaying images. Move the mouse over each image to check RGB values.")
        viewer.show()

    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
