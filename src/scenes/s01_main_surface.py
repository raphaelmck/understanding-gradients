from manim import *
import numpy as np

# ---------------------------
# Shared helpers (fast reuse)
# ---------------------------

def peaks_f(u, v):
    return (
        3 * (1 - u) ** 2 * np.exp(-u**2 - (v + 1) ** 2)
        - 10 * (u / 5 - u**3 - v**5) * np.exp(-u**2 - v**2)
        - (1 / 3) * np.exp(-(u + 1) ** 2 - v**2)
    )

def grad_numeric(f, u, v, eps=1e-3):
    fu = (f(u + eps, v) - f(u - eps, v)) / (2 * eps)
    fv = (f(u, v + eps) - f(u, v - eps)) / (2 * eps)
    return fu, fv

def build_world():
    axes = ThreeDAxes(
        x_range=[-3, 3, 1],
        y_range=[-3, 3, 1],
        z_range=[-8, 8, 2],
        x_length=7,
        y_length=7,
        z_length=5,
    )
    axes_labels = axes.get_axis_labels(
        Tex("x").scale(0.7),
        Tex("y").scale(0.7),
        Tex("z").scale(0.7),
    )

    def c2p(u, v):
        return axes.c2p(u, v, peaks_f(u, v))

    surface = Surface(
        lambda u, v: c2p(u, v),
        u_range=[-3, 3],
        v_range=[-3, 3],
        resolution=(64, 64),
        should_make_jagged=False,
    )
    surface.set_style(fill_opacity=0.95, stroke_width=0)
    surface.set_fill_by_value(
            axes=axes,
            colors=[(BLUE_E, -8), (BLUE_C, 0), (GREEN_C, 4), (YELLOW_E, 8)],
            axis=2,
        )
    surface.set_style(fill_opacity=0.90, stroke_width=0)  # set this AFTER fill_by_value


    return axes, axes_labels, surface

def tangent_arrow(axes, u0, v0, z0, fu, fv, dx, dy, length=0.9, color=BLUE_A, thickness=0.02):
    d = np.array([dx, dy], dtype=float)
    d /= np.linalg.norm(d)
    du, dv = length * d[0], length * d[1]

    start = axes.c2p(u0, v0, z0)
    end = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv)
    return Arrow3D(start, end, thickness=thickness, color=color)

def dir_slope(fu, fv, dx, dy):
    d = np.array([dx, dy], dtype=float)
    d /= np.linalg.norm(d)
    return fu * d[0] + fv * d[1]

def build_surface(axes):
    def c2p(u, v):
        return axes.c2p(u, v, peaks_f(u, v))

    surface = Surface(
        lambda u, v: c2p(u, v),
        u_range=[-3, 3],
        v_range=[-3, 3],
        resolution=(64, 64),
        should_make_jagged=False,
    )
    surface.set_style(fill_opacity=0.95, stroke_width=0)
    surface.set_fill_by_value(
        axes=axes,
        colors=[(BLUE_D, -8), (BLUE_B, 0), (GREEN_B, 4), (YELLOW, 8)],
        axis=2,
    )
    return surface

# ---------------------------
# Scene 1: Surface reveal
# ---------------------------

class S01_SurfaceReveal(ThreeDScene):
    def construct(self):
        # 1. Setup (We need the axes for calculations, but won't add them yet)
        axes, axes_labels, static_surface = build_world()
        
        # ---------------------------------------------------------
        # Syncing with Scene 2 Start
        # Scene 2 starts at: phi=72, theta=25, zoom=1.02
        # ---------------------------------------------------------
        phi    = 72 * DEGREES
        theta0 = -35 * DEGREES
        theta1 = 25 * DEGREES   # Matches Scene 2 start
        zoom0  = 0.90
        zoom1  = 1.02           # Matches Scene 2 start
        
        self.set_camera_orientation(phi=phi, theta=theta0, zoom=zoom0)

        # 2. Trackers
        # We track 'u' for the surface and 'alpha' for the camera
        u_max_tracker = ValueTracker(-2.99)
        cam_tracker   = ValueTracker(0.0)

        # 3. Dynamic Surface
        def get_dynamic_surface():
            u_current = u_max_tracker.get_value()
            
            # Dynamic Resolution Calculation:
            # We want the resolution to match the final static surface (64) exactly
            # when the animation finishes, to prevent a "pop" in geometry.
            total_u_len = 6.0 # from -3 to 3
            current_len = u_current - (-3)
            
            # Calculate proportional resolution
            # If current_len is 3 (halfway), res is 32. At end, res is 64.
            u_res = int((current_len / total_u_len) * 64)
            u_res = max(4, u_res) # clamp minimum to avoid errors
            
            surface = Surface(
                lambda u, v: axes.c2p(u, v, peaks_f(u, v)),
                u_range=[-3, u_current],
                v_range=[-3, 3],
                resolution=(u_res, 64), # Match v-res of static surface
            )
            
            # -----------------------------------------------------
            # CRITICAL: Match Style to Scene 2
            # Scene 2's static surface has fill_opacity=0.90
            # -----------------------------------------------------
            surface.set_style(fill_opacity=0.90, stroke_width=0)
            
            surface.set_fill_by_value(
                axes=axes,
                colors=[(BLUE_E, -8), (BLUE_C, 0), (GREEN_C, 4), (YELLOW_E, 8)],
                axis=2,
            )
            return surface

        dynamic_surface = always_redraw(get_dynamic_surface)

        # 4. Scanner Line (Optional visual flair)
        # We fade this out right at the end to blend perfectly into static mesh
        scanner_opacity = ValueTracker(0.8)

        def get_scanner_line():
            u_val = u_max_tracker.get_value()
            opac = scanner_opacity.get_value()
            
            if u_val <= -2.9: return VGroup()
            
            line = ParametricFunction(
                lambda v: axes.c2p(u_val, v, peaks_f(u_val, v)),
                t_range=[-3, 3],
                color=WHITE
            )
            line.set_stroke(width=2, opacity=opac)
            return line

        scanner_line = always_redraw(get_scanner_line)

        self.add(dynamic_surface, scanner_line)

        # 5. Camera Updater (Smooth 3D motion)
        def update_camera(mob):
            alpha = cam_tracker.get_value()
            theta = interpolate(theta0, theta1, alpha)
            zoom  = interpolate(zoom0, zoom1, alpha)
            self.set_camera_orientation(phi=phi, theta=theta, zoom=zoom)

        dummy_cam = Mobject()
        dummy_cam.add_updater(update_camera)
        self.add(dummy_cam)

        # 6. Animation
        # We slow it down to 8 seconds for a smooth feel
        self.play(
            u_max_tracker.animate.set_value(3.0),
            cam_tracker.animate.set_value(1.0),
            run_time=8.0,
            rate_func=smooth
        )

        # Fade out scanner line quickly at the very end so it doesn't get stuck
        self.play(scanner_opacity.animate.set_value(0.0), run_time=0.5)

        # 7. Seamless Handoff
        # Remove dynamic elements and add static one
        dummy_cam.remove_updater(update_camera)
        self.remove(dummy_cam, dynamic_surface, scanner_line)
        self.add(static_surface)
        
        self.wait(1.0)

class S01_5_Transition(ThreeDScene):
    def construct(self):
        axes, axes_labels, surface = build_world()

        # 1. Match End of Scene 1
        self.set_camera_orientation(phi=72 * DEGREES, theta=25 * DEGREES, zoom=1.02)
        self.add(surface)

        # 2. Camera Move to Target
        u0, v0 = 1.2, 1.0
        z0 = peaks_f(u0, v0)
        p_focus = axes.c2p(u0, v0, z0)

        self.move_camera(
            theta=15 * DEGREES,
            zoom=1.4,
            focal_point=p_focus,
            run_time=2.5,
            rate_func=smooth
        )

        # 3. Ripple Effect
        radius_tracker = ValueTracker(0.0)
        opacity_tracker = ValueTracker(1.0)

        def get_ripple():
            r = radius_tracker.get_value()
            opac = opacity_tracker.get_value()
            if r < 0.01: return VGroup()
            
            ring = ParametricFunction(
                lambda t: axes.c2p(
                    u0 + r * np.cos(t), 
                    v0 + r * np.sin(t), 
                    peaks_f(u0 + r * np.cos(t), v0 + r * np.sin(t))
                ),
                t_range=[0, TAU],
                color=WHITE
            )
            ring.set_stroke(width=3, opacity=opac)
            return ring

        ripple = always_redraw(get_ripple)
        dot = Dot3D(p_focus, radius=0.05, color=WHITE)

        self.add(ripple, dot)
        self.play(FadeIn(dot, scale=0.5), run_time=0.3)
        
        self.play(
            radius_tracker.animate.set_value(0.6),
            opacity_tracker.animate.set_value(0.0),
            run_time=1.5,
            rate_func=linear
        )
        self.remove(ripple)

        # 4. Minimalist Directional Arrows
        fu, fv = grad_numeric(peaks_f, u0, v0)

        directions = [
            (1, 0),   # Right
            (0, -1),  # Down
            (-1, 0),  # Left
            (0, 1)    # Up 
        ]

        arrows = []
        for (dx, dy) in directions:
            arr = tangent_arrow(
                axes, u0, v0, z0, fu, fv, dx, dy, 
                length=0.8,         # Uniform size
                color=RED, 
                thickness=0.015     # Thin
            )
            arrows.append(arr)

        self.play(
            LaggedStart(
                *[GrowFromPoint(arr, p_focus) for arr in arrows],
                lag_ratio=0.2
            ),
            run_time=1.2
        )
        
        self.wait(0.5)

        # 5. Fade Out ONLY Dot and Arrows (Surface stays)
        self.play(
            FadeOut(dot),
            *[FadeOut(arr) for arr in arrows],
            run_time=1.5
        )
        self.wait(0.5)

# ---------------------------
# Scene 2: Axes + point + height
# ---------------------------

class S02_PointAndHeight(ThreeDScene):
    def construct(self):
        axes, axes_labels, surface = build_world()

        # -----------------------------------------------------
        # 1. Exact Match with Scene 1.5 End State
        # -----------------------------------------------------
        # The camera ended looking at the previous point (1.2, 1.0)
        u_prev, v_prev = 1.2, 1.0
        z_prev = peaks_f(u_prev, v_prev)
        p_prev_focus = axes.c2p(u_prev, v_prev, z_prev)

        # Set initial camera to match Scene 1.5 exactly
        self.set_camera_orientation(
            phi=72 * DEGREES, 
            theta=15 * DEGREES, 
            zoom=1.4, 
            focal_point=p_prev_focus
        )

        self.add(surface)

        # -----------------------------------------------------
        # 2. Contextual Reveal (Axes & Floor)
        # -----------------------------------------------------
        # Create floor/axes invisible first
        floor = Surface(
            lambda u, v: axes.c2p(u, v, -0.02),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(2, 2),
        )
        floor.set_style(fill_opacity=0.20, fill_color=GRAY_D, stroke_width=0)
        
        # Fade them in while we are still zoomed in on the previous spot
        self.play(
            FadeIn(floor),
            Create(axes),
            run_time=1.5
        )
        self.play(FadeIn(axes_labels), run_time=0.5)

        # -----------------------------------------------------
        # 3. The "Pull Back" (Context)
        # -----------------------------------------------------
        # Pull back to see the whole graph and center the view
        self.move_camera(
            phi=65 * DEGREES,
            theta=-35 * DEGREES,
            zoom=0.8,
            focal_point=axes.c2p(0, 0, 0), # Reset focus to Origin
            run_time=2.5,
            rate_func=smooth
        )

        # -----------------------------------------------------
        # 4. Highlight the New Point
        # -----------------------------------------------------
        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)

        p_ground  = axes.c2p(u0, v0, 0)
        p_surface = axes.c2p(u0, v0, z0)

        ground_dot  = Dot3D(p_ground, radius=0.06, color=WHITE)
        surf_dot    = Dot3D(p_surface, radius=0.07, color=WHITE)
        height_line = DashedLine(p_ground, p_surface, dash_length=0.08).set_color(GRAY_B)
        
        val_label = DecimalNumber(z0, num_decimal_places=2).scale(0.5).set_color(WHITE)
        val_label.next_to(surf_dot, UR, buff=0.15)

        # Dim surface briefly to show the height line clearly
        self.play(surface.animate.set_style(fill_opacity=0.25, stroke_width=0), run_time=0.8)

        self.play(FadeIn(ground_dot), run_time=0.3)
        self.play(Create(height_line), run_time=0.6)
        self.play(FadeIn(surf_dot), FadeIn(val_label), run_time=0.5)

        # Restore surface opacity
        self.play(surface.animate.set_style(fill_opacity=0.95, stroke_width=0), run_time=0.8)

        # -----------------------------------------------------
        # 5. Final Move: Zoom onto the Point (Setup for Derivative)
        # -----------------------------------------------------
        # Voiceover: "Now what we want is local steepest ascent: at one point..."
        # We zoom in very close to p_surface to isolate the "slope"
        
        self.move_camera(
            phi=55 * DEGREES,       # Slightly lower angle to see the "hill"
            theta=-15 * DEGREES,    # Rotate to a clear profile view of the slope
            zoom=1.9,               # Significant zoom in
            focal_point=p_surface,  # Focus EXACTLY on the dot
            run_time=3.0,
            rate_func=smooth
        )
        
        self.wait(1.0)

# ---------------------------
# Scene 3: Tangent patch + many slopes
# ---------------------------

class S03_DirectionalSlopes(ThreeDScene):
    def construct(self):
        axes, axes_labels, surface = build_world()
        self.set_camera_orientation(phi=68 * DEGREES, theta=-20 * DEGREES, zoom=1.25)

        # Keep axes optional here; I like them faint
        axes.set_opacity(0.65)
        axes_labels.set_opacity(0.65)

        self.add(surface, axes, axes_labels)

        # Point
        u0, v0 = 0.9, 0.8
        z0 = peaks_f(u0, v0)
        p_surface = axes.c2p(u0, v0, z0)
        surf_dot = Dot3D(p_surface, radius=0.07, color=WHITE)
        self.add(surf_dot)

        # Tangent plane patch
        fu, fv = grad_numeric(peaks_f, u0, v0)

        plane = Surface(
            lambda a, b: axes.c2p(a, b, z0 + fu * (a - u0) + fv * (b - v0)),
            u_range=[u0 - 0.75, u0 + 0.75],
            v_range=[v0 - 0.75, v0 + 0.75],
            resolution=(10, 10),
        )
        plane.set_style(fill_opacity=0.25, fill_color=WHITE, stroke_width=0)

        self.play(FadeIn(plane), run_time=0.7)

        # A few directions + labels
        dirs = [(1, 0), (0.4, 1), (-1, 0.2), (0.2, -1)]
        arrows = VGroup()
        labels = VGroup()

        for (dx, dy) in dirs:
            arr = tangent_arrow(axes, u0, v0, z0, fu, fv, dx, dy, length=0.85, color=BLUE_B)
            s = dir_slope(fu, fv, dx, dy)
            lab = DecimalNumber(s, num_decimal_places=2).scale(0.35).set_color(WHITE)
            lab.move_to(arr.get_end() + 0.18 * UP)
            arrows.add(arr)
            labels.add(lab)

        # Camera drift while arrows appear
        self.play(
            AnimationGroup(
                LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.12),
                self.camera.frame.animate.set_width(self.camera.frame.get_width() * 0.94),
                run_time=1.2,
            )
        )
        self.play(FadeIn(labels), run_time=0.5)

        # Gentle orbit around the patch (tiny, not distracting)
        self.move_camera(theta=-8 * DEGREES, run_time=1.0)  # relative feel
        self.wait(0.8)

# ---------------------------
# Scene 4: Collapse to gradient + rotating direction + dot product HUD
# ---------------------------

class S04_GradientOverlap(ThreeDScene):
    def construct(self):
        axes, axes_labels, surface = build_world()
        self.set_camera_orientation(phi=68 * DEGREES, theta=-20 * DEGREES, zoom=1.25)

        axes.set_opacity(0.6)
        axes_labels.set_opacity(0.6)
        self.add(surface, axes, axes_labels)

        u0, v0 = 0.9, 0.8
        z0 = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)

        # Dot + tangent patch
        p_surface = axes.c2p(u0, v0, z0)
        surf_dot = Dot3D(p_surface, radius=0.07, color=WHITE)

        plane = Surface(
            lambda a, b: axes.c2p(a, b, z0 + fu * (a - u0) + fv * (b - v0)),
            u_range=[u0 - 0.75, u0 + 0.75],
            v_range=[v0 - 0.75, v0 + 0.75],
            resolution=(10, 10),
        )
        plane.set_style(fill_opacity=0.25, fill_color=WHITE, stroke_width=0)

        self.add(plane, surf_dot)

        # Gradient direction
        gxy = np.array([fu, fv], dtype=float)
        gxy_norm = np.linalg.norm(gxy)
        g_dir = np.array([1.0, 0.0]) if gxy_norm < 1e-9 else (gxy / gxy_norm)

        grad_arrow = tangent_arrow(axes, u0, v0, z0, fu, fv, g_dir[0], g_dir[1], length=1.05, color=YELLOW, thickness=0.03)
        grad_label = MathTex(r"\nabla f").scale(0.6).set_color(YELLOW)
        grad_label.move_to(grad_arrow.get_end() + 0.22 * UP)

        # Start with a few arrows then fade them out into gradient
        dirs = [(1, 0), (0.4, 1), (-1, 0.2), (0.2, -1)]
        arrows = VGroup(*[
            tangent_arrow(axes, u0, v0, z0, fu, fv, dx, dy, length=0.8, color=BLUE_B, thickness=0.02)
            for (dx, dy) in dirs
        ])

        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.12), run_time=0.8)
        self.play(FadeOut(arrows), Create(grad_arrow), FadeIn(grad_label), run_time=1.0)

        # Camera: subtle push-in at the moment gradient appears
        self.move_camera(phi=64 * DEGREES, theta=-12 * DEGREES, zoom=1.38, run_time=1.0)

        # Rotating direction arrow + HUD
        theta = ValueTracker(0.0)

        def rotating_arrow():
            ang = theta.get_value()
            d = np.array([np.cos(ang), np.sin(ang)])
            return tangent_arrow(axes, u0, v0, z0, fu, fv, d[0], d[1], length=0.95, color=BLUE_A, thickness=0.02)

        rot_arr = always_redraw(rotating_arrow)

        value = DecimalNumber(0.0, num_decimal_places=2).scale(0.55).set_color(WHITE)
        value_label = MathTex(r"D_{\mathbf{v}}f \approx \nabla f \cdot \mathbf{v}").scale(0.55).set_color(GRAY_A)

        hud = VGroup(value_label, value).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UL, buff=0.4)
        self.add_fixed_in_frame_mobjects(hud)

        def update_value(m):
            ang = theta.get_value()
            m.set_value(fu * np.cos(ang) + fv * np.sin(ang))
            return m

        value.add_updater(update_value)

        self.play(FadeIn(hud), FadeIn(rot_arr), run_time=0.6)

        # Rotation
        self.play(theta.animate.set_value(TAU), run_time=3.6, rate_func=linear)

        # Perpendicular ~ 0
        g_ang = np.arctan2(g_dir[1], g_dir[0])
        self.play(theta.animate.set_value(g_ang + PI / 2), run_time=1.0)
        self.wait(0.35)

        # Align with gradient = max
        self.play(theta.animate.set_value(g_ang), run_time=1.0)
        self.wait(1.0)
