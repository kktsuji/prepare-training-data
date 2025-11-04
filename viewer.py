import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor


class RGBImageViewer:
    def __init__(self, image_path):
        # Load image (OpenCV uses BGR, so convert to RGB)
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image file: {image_path}")

        # BGR → RGB conversion
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Get image height and width
        self.height, self.width = self.image.shape[:2]

        # Matplotlib display settings
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.im = self.ax.imshow(self.image)
        self.ax.set_title(
            "RGB Image Viewer - Move mouse over image to check RGB values"
        )

        # Text for displaying RGB values
        self.text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set up mouse event
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Display cursor
        self.cursor = Cursor(self.ax, useblit=True, color="red", linewidth=1)

    def on_mouse_move(self, event):
        """Event handler for mouse movement"""
        if event.inaxes != self.ax:
            return

        # Convert mouse coordinates to image coordinates
        x, y = int(event.xdata), int(event.ydata)

        # Check if coordinates are within image bounds
        if 0 <= x < self.width and 0 <= y < self.height:
            # Get RGB values
            r, g, b = self.image[y, x]

            # Display RGB values
            rgb_text = f"Coordinates: ({x}, {y})\nRGB: ({r}, {g}, {b})\nHex: #{r:02x}{g:02x}{b:02x}"
            self.text.set_text(rgb_text)

            # Update display
            self.fig.canvas.draw_idle()

    def show(self):
        """Display the image"""
        plt.tight_layout()
        plt.show()


def main():
    # Please specify the path to the image file
    # image_path = input("Enter the image file path: ")
    base_path = "../Data/ai-gen-image-stats/"
    fp = "original/ctcs-resized-bilinear-512/160606-L-4-2-tif_img-no150_y525_x38.png"
    fp = "lora-trained-linear/final-model-degamma/seed8_rsd_lora.png"
    # fp = "lora-trained-linear-resized/final-model_degamma/seed2_rsd_lora.png"
    # fp = "lora-trained-gamma/ctcs-lora-degamma/seed16_rsd_lora.png"
    image_path = base_path + fp

    try:
        # Create an instance of RGBImageViewer
        viewer = RGBImageViewer(image_path)

        # Display the image
        viewer.show()

    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
