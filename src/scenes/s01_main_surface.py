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
            z = (3 * (1 - u)**2 * np.exp(-u**2 - (v + 1)**2) 
                 - 10 * (u / 5 - u**3 - v**5) * np.exp(-u**2 - v**2) 
                 - 1 / 3 * np.exp(-(u + 1)**2 - v**2))
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
        surface.set_style(fill_opacity=0.95, stroke_width=0)
        surface.set_fill_by_value(
            axes=axes,
            colors=[(BLUE_D, -8), (BLUE_B, 0), (GREEN_B, 4), (YELLOW, 8)],
            axis=2  # z-axis
        )

        # 4. Animation
        self.add(axes)
        self.play(Create(surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.1) # Rotate gently
        self.wait(10)
