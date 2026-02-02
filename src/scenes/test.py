from manim import *
import numpy as np

class LossSurface(ThreeDScene):
    def construct(self):
        # 1. Set up the camera and axes
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-8, 8, 2],
            x_length=7,
            y_length=7,
            z_length=5
        ).add_coordinates()

        # 2. Define the "Peaks" function
        def peaks_func(u, v):
            # This is the exact MATLAB peaks equation
            z = 0.5 * (u**2 + v**2) - np.cos(5*u) - np.cos(5*v)
            return axes.c2p(u, v, z)

        # 3. Create the Surface object
        surface = Surface(
            peaks_func,
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(64, 64), # Higher = smoother but slower
            should_make_jagged=False
        )
        
        # Style the surface (blue-ish style typical of these plots)
        surface.set_style(fill_opacity=0.8, fill_color=BLUE)
        surface.set_fill_by_checkerboard(BLUE, DARK_BLUE, opacity=0.8)

        # 4. Animation
        self.add(axes)
        self.play(Create(surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.1) # Rotate gently
        self.wait(3)
