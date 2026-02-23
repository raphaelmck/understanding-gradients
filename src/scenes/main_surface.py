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

class S02_5_DerivativeOverlay(Scene):
    def construct(self):
        # -----------------------------------------------------
        # Setup: Transparent Background
        # -----------------------------------------------------
        # Render with '-t' flag: manim -ql -t file.py S02_5_DerivativeOverlay
        self.camera.background_color = "#00000000"

        # 1. Define Axes
        axes = Axes(
            x_range=[-1, 6], 
            y_range=[-1, 5], 
            x_length=8, 
            y_length=5,
            axis_config={"color": GRAY_C, "stroke_width": 2, "include_tip": True}
        ).to_edge(DOWN, buff=1.0)

        # 2. Define a "Curvier" Function
        # y = 1.5 * sin(0.8x) + 2
        # This provides a nice tall hill shape
        def func_val(x):
            return 1.5 * np.sin(0.8 * x) + 2.0

        graph = axes.plot(func_val, x_range=[-0.5, 5.5], color=BLUE, stroke_width=5)
        graph_label = axes.get_graph_label(graph, label="f(x)", x_val=5.5, direction=RIGHT)

        # 3. Define Points (Far apart to avoid intersection)
        
        # Point 1: Ascent (Left side)
        x1 = 0.8
        y1 = func_val(x1)
        p1 = axes.c2p(x1, y1)
        dot1 = Dot(p1, color=GREEN, radius=0.12)
        
        # Point 2: Descent (Right side)
        x2 = 3.5
        y2 = func_val(x2)
        p2 = axes.c2p(x2, y2)
        dot2 = Dot(p2, color=RED, radius=0.12)

        # 4. Tangent Line Helper
        def create_tangent(x, point, color):
            # Calculate angle numerically
            dt = 0.001
            p_curr = axes.c2p(x, func_val(x))
            p_next = axes.c2p(x + dt, func_val(x + dt))
            angle = angle_of_vector(p_next - p_curr)
            
            # Create line centered on point
            line = Line(LEFT, RIGHT, color=color, stroke_width=4).set_length(2.5)
            line.rotate(angle)
            line.move_to(point)
            return line

        tangent1 = create_tangent(x1, p1, GREEN)
        tangent2 = create_tangent(x2, p2, RED)

        # 5. Labels
        label1 = Text("Slope > 0", font_size=28, color=GREEN).next_to(tangent1, UP+LEFT, buff=0.2)
        label2 = Text("Slope < 0", font_size=28, color=RED).next_to(tangent2, UP+RIGHT, buff=0.2)

        # Grouping for clean fade out later
        full_group = VGroup(axes, graph, graph_label, dot1, dot2, tangent1, tangent2, label1, label2)

        # -----------------------------------------------------
        # Animation Sequence (Spaced Out)
        # -----------------------------------------------------

        # 1. Draw the Graph
        self.play(DrawBorderThenFill(axes), run_time=1.0)
        self.play(Create(graph), Write(graph_label), run_time=1.5)
        
        # Pause to appreciate the curve
        self.wait(1.0)

        # 2. Show Ascent (Up Slope)
        self.play(GrowFromCenter(dot1), run_time=0.5)
        self.play(Create(tangent1), run_time=0.8)
        self.play(Write(label1), run_time=0.6)

        # Distinct Pause ("Not just a single slope...")
        self.wait(1.5)

        # 3. Show Descent (Down Slope)
        self.play(GrowFromCenter(dot2), run_time=0.5)
        self.play(Create(tangent2), run_time=0.8)
        self.play(Write(label2), run_time=0.6)

        # Final Pause for voiceover wrap-up
        self.wait(2.0)

        # 4. Clean Fade Out
        self.play(FadeOut(full_group), run_time=1.0)

class S02_7_ZoomToPoint(ThreeDScene):
    """Fade Scene 2 helpers + smooth zoom to point, all in one scene."""
    def construct(self):
        axes, axes_labels, surface = build_world()

        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)

        p_ground  = axes.c2p(u0, v0, 0)
        p_surface = axes.c2p(u0, v0, z0)

        floor = Surface(
            lambda u, v: axes.c2p(u, v, -0.02),
            u_range=[-3, 3], v_range=[-3, 3], resolution=(2, 2),
        )
        floor.set_style(fill_opacity=0.20, fill_color=GRAY_D, stroke_width=0)

        ground_dot  = Dot3D(p_ground, radius=0.06, color=WHITE)
        surf_dot    = Dot3D(p_surface, radius=0.07, color=WHITE)
        height_line = DashedLine(p_ground, p_surface, dash_length=0.08).set_color(GRAY_B)
        val_label   = DecimalNumber(z0, num_decimal_places=2).scale(0.5).set_color(WHITE)
        val_label.next_to(surf_dot, UR, buff=0.15)

        # Start UNSHIFTED — exact match with S02 end frame
        self.set_camera_orientation(
            phi=55 * DEGREES,
            theta=-15 * DEGREES,
            zoom=1.9,
            focal_point=p_surface,
        )
        surface.set_style(fill_opacity=0.95, stroke_width=0)
        self.add(surface, axes, axes_labels, floor, ground_dot, height_line, surf_dot, val_label)

        self.wait(0.3)

        # Fade out helpers + shrink dot + dim surface
        self.play(
            FadeOut(axes), FadeOut(axes_labels), FadeOut(floor),
            FadeOut(ground_dot), FadeOut(height_line), FadeOut(val_label),
            surface.animate.set_style(fill_opacity=0.85, stroke_width=0),
            surf_dot.animate.scale(0.05 / 0.07),
            run_time=1.5,
            rate_func=smooth,
        )

        # Invisible shift: move world so point is at ORIGIN and swap focal_point.
        # With helpers gone there are no reference lines, so the shift is imperceptible.
        shift_vec = -p_surface
        surface.shift(shift_vec)
        surf_dot.shift(shift_vec)
        self.set_camera_orientation(
            phi=55 * DEGREES,
            theta=-15 * DEGREES,
            zoom=1.9,
            focal_point=ORIGIN,
        )

        # Smooth zoom — works perfectly because focal_point is ORIGIN
        self.move_camera(
            phi=70 * DEGREES,
            theta=-10 * DEGREES,
            zoom=3.0,
            focal_point=ORIGIN,
            run_time=2.5,
            rate_func=smooth,
        )

        self.wait(0.5)


class S03_InfiniteSlopes(ThreeDScene):
    def construct(self):
        axes, axes_labels, surface = build_world()

        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)

        # Shift world — matches S02_7 end
        world_offset = axes.c2p(u0, v0, z0)
        VGroup(axes, surface).shift(-world_offset)
        p_focus = ORIGIN

        # Matches S02_7 end camera exactly
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-10 * DEGREES,
            zoom=3.0,
            focal_point=ORIGIN,
        )

        surface.set_style(fill_opacity=0.85, stroke_width=0)
        dot = Dot3D(ORIGIN, radius=0.05, color=WHITE)
        self.add(surface, dot)

        self.wait(0.3)

        # -----------------------------------------------------
        # 3. Infinite Slopes (Clockwise from Downhill)
        # -----------------------------------------------------
        max_slope = np.linalg.norm([fu, fv])
        g_ang = np.arctan2(fv, fu)

        num_arrows = 12
        start_angle = g_ang + PI
        angles_to_show = [start_angle - i * (TAU / num_arrows) for i in range(num_arrows)]

        z_bump = OUT * 0.04         # lift arrows/trails above surface (OUT = z-axis)
        trail_len = 0.25
        arr_len = 0.25

        # Trail factory (shared by all arrows)
        def make_trail_builder(dx_v, dy_v, head_tr, streak_len):
            _bump = z_bump.copy()
            def _build():
                h = head_tr.get_value()
                t = max(0.0, h - streak_len)
                if h < 0.005:
                    return VGroup()
                opac = max(0.0, 1.0 - h / 1.5)
                if opac < 0.01:
                    return VGroup()
                trail = ParametricFunction(
                    lambda s, _dx=dx_v, _dy=dy_v: axes.c2p(
                        u0 + s * _dx, v0 + s * _dy,
                        peaks_f(u0 + s * _dx, v0 + s * _dy),
                    ) + _bump,
                    t_range=[t, h],
                    color=WHITE,
                    stroke_width=2.5,
                    stroke_opacity=opac,
                )
                glow = trail.copy().set_stroke(
                    width=5, color=WHITE, opacity=opac * 0.3
                )
                return VGroup(glow, trail)
            return _build

        # --- Pre-build all arrows, trackers, and trail snakes ---
        arrows = []
        trackers = []
        snakes = []

        for angle in angles_to_show:
            dx, dy = np.cos(angle), np.sin(angle)
            slope = fu * dx + fv * dy
            norm_slope = slope / max_slope

            if norm_slope > 0:
                col = interpolate_color(YELLOW, RED, min(norm_slope * 1.2, 1.0))
            else:
                col = interpolate_color(TEAL, BLUE, min(-norm_slope * 1.2, 1.0))

            d_hat = np.array([dx, dy], dtype=float)
            d_hat /= np.linalg.norm(d_hat)
            du, dv = arr_len * d_hat[0], arr_len * d_hat[1]

            start_pt = axes.c2p(u0, v0, z0) + z_bump
            end_pt = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv) + z_bump

            arr = Arrow3D(
                start=start_pt,
                end=end_pt,
                thickness=0.004,
                height=0.07,
                base_radius=0.02,
                color=col,
            )

            tracker = ValueTracker(0.001)
            snake = always_redraw(make_trail_builder(dx, dy, tracker, trail_len))

            arrows.append(arr)
            trackers.append(tracker)
            snakes.append(snake)

        # Add all trail snakes up-front (invisible: tracker ≈ 0)
        self.add(*snakes)

        # Per-arrow sequence: fast grow + slow trail, then dim
        sequences = []
        for arr, tracker in zip(arrows, trackers):
            sequences.append(
                Succession(
                    AnimationGroup(
                        GrowFromPoint(arr, p_focus, run_time=0.25),
                        tracker.animate(run_time=1.2, rate_func=linear).set_value(1.5),
                    ),
                    arr.animate(run_time=0.25).set_opacity(0.15),
                )
            )

        # Play all overlapping via LaggedStart
        self.play(LaggedStart(*sequences, lag_ratio=0.35))

        # Clean up trail redraws
        self.remove(*snakes)

        self.wait(2.0)


class S04_GradientReveal(ThreeDScene):
    """All directional arrows rotate/move into the single gradient vector."""
    def construct(self):
        axes, axes_labels, surface = build_world()

        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)

        # Shift world — same as S03
        world_offset = axes.c2p(u0, v0, z0)
        VGroup(axes, surface).shift(-world_offset)

        # Camera: exact S03 end
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-10 * DEGREES,
            zoom=3.0,
            focal_point=ORIGIN,
        )

        surface.set_style(fill_opacity=0.85, stroke_width=0)
        dot = Dot3D(ORIGIN, radius=0.05, color=WHITE)
        self.add(surface, dot)

        # --- Recreate the 12 dimmed arrows (S03 end state) ---
        max_slope = np.linalg.norm([fu, fv])
        g_ang = np.arctan2(fv, fu)
        num_arrows = 12
        start_angle = g_ang + PI
        angles_to_show = [start_angle - i * (TAU / num_arrows) for i in range(num_arrows)]

        z_bump = OUT * 0.04             # lift above surface (OUT = z-axis)
        arr_len = 0.25

        # Extra lift for the gradient arrow tip to clear the surface
        z_bump_grad = OUT * 0.12

        dim_arrows = VGroup()
        for angle in angles_to_show:
            dx, dy = np.cos(angle), np.sin(angle)
            slope = fu * dx + fv * dy
            norm_slope = slope / max_slope

            if norm_slope > 0:
                col = interpolate_color(YELLOW, RED, min(norm_slope * 1.2, 1.0))
            else:
                col = interpolate_color(TEAL, BLUE, min(-norm_slope * 1.2, 1.0))

            d_hat = np.array([dx, dy], dtype=float)
            d_hat /= np.linalg.norm(d_hat)
            du, dv = arr_len * d_hat[0], arr_len * d_hat[1]

            start_pt = axes.c2p(u0, v0, z0) + z_bump
            end_pt = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv) + z_bump

            arr = Arrow3D(
                start=start_pt,
                end=end_pt,
                thickness=0.004,
                height=0.07,
                base_radius=0.02,
                color=col,
            ).set_opacity(0.15)
            dim_arrows.add(arr)

        self.add(dim_arrows)
        dim_arrows.set_z_index(1)
        dot.set_z_index(2)
        surface.set_z_index(-1)
        self.wait(0.3)

        # --- Build the gradient arrow (target for merge) ---
        grad_dir = np.array([fu, fv], dtype=float)
        grad_mag = np.linalg.norm(grad_dir)
        grad_hat = grad_dir / grad_mag
        grad_len = 0.25                    # same length as directional arrows
        gdu, gdv = grad_len * grad_hat[0], grad_len * grad_hat[1]

        grad_start = axes.c2p(u0, v0, z0) + z_bump_grad
        grad_end = axes.c2p(
            u0 + gdu, v0 + gdv,
            z0 + fu * gdu + fv * gdv,
        ) + z_bump_grad

        # Build a "target" copy of the gradient arrow for each dim arrow
        # so they all move/rotate to the same place
        target_arrows = VGroup()
        for _ in range(num_arrows):
            t_arr = Arrow3D(
                start=grad_start,
                end=grad_end,
                thickness=0.004,
                height=0.07,
                base_radius=0.02,
                color=RED,
            ).set_opacity(0.6)
            target_arrows.add(t_arr)

        # --- Step 1: Swing camera for a nicer gradient view ---
        self.play(
            dim_arrows.animate.set_opacity(0.5),
            run_time=0.4,
            rate_func=smooth,
        )

        self.move_camera(
            phi=60 * DEGREES,
            theta=-35 * DEGREES,
            zoom=3.2,
            focal_point=ORIGIN,
            run_time=2.0,
            rate_func=smooth,
        )

        # --- Step 2: All arrows rotate/move into the gradient direction ---
        move_anims = []
        for dim_arr, tgt_arr in zip(dim_arrows, target_arrows):
            move_anims.append(Transform(dim_arr, tgt_arr))

        self.play(
            LaggedStart(*move_anims, lag_ratio=0.06),
            run_time=3.0,
            rate_func=smooth,
        )

        # --- Step 3: Collapse into the final bold gradient arrow ---
        grad_arrow = Arrow3D(
            start=grad_start,
            end=grad_end,
            thickness=0.012,
            height=0.10,
            base_radius=0.035,
            color=RED,
        )
        # Add a glow duplicate for extra visibility
        grad_glow = grad_arrow.copy().set_color(RED_A).set_opacity(0.35)

        # Keep arrows rendered on top of surface
        surface.set_z_index(-1)
        grad_arrow.set_z_index(1)
        grad_glow.set_z_index(1)

        self.play(
            FadeOut(dim_arrows),
            FadeIn(grad_arrow),
            FadeIn(grad_glow),
            run_time=1.2,
            rate_func=smooth,
        )

        # --- Step 4: Slow rotate around the gradient to show it off ---
        self.move_camera(
            theta=-55 * DEGREES,
            focal_point=ORIGIN,
            run_time=2.5,
            rate_func=smooth,
        )

        self.wait(2.0)


class S05_RotatingSlope(ThreeDScene):
    """A direction arrow rotates; a camera-facing number shows the slope."""
    def construct(self):
        axes, axes_labels, surface = build_world()

        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)

        # Shift world — same as S04
        world_offset = axes.c2p(u0, v0, z0)
        VGroup(axes, surface).shift(-world_offset)

        # Camera: exact S04 end
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-55 * DEGREES,
            zoom=3.2,
            focal_point=ORIGIN,
        )

        surface.set_style(fill_opacity=0.85, stroke_width=0)
        surface.set_z_index(-1)
        dot = Dot3D(ORIGIN, radius=0.05, color=WHITE)
        dot.set_z_index(2)
        self.add(surface, dot)

        # --- Gradient arrow (matches S04 end) ---
        z_bump_grad = OUT * 0.12
        grad_dir = np.array([fu, fv], dtype=float)
        grad_mag = np.linalg.norm(grad_dir)
        grad_hat = grad_dir / grad_mag
        g_ang = np.arctan2(fv, fu)
        grad_len = 0.25
        gdu, gdv = grad_len * grad_hat[0], grad_len * grad_hat[1]

        grad_start = axes.c2p(u0, v0, z0) + z_bump_grad
        grad_end = axes.c2p(
            u0 + gdu, v0 + gdv,
            z0 + fu * gdu + fv * gdv,
        ) + z_bump_grad

        grad_arrow = Arrow3D(
            start=grad_start, end=grad_end,
            thickness=0.012, height=0.10, base_radius=0.035,
            color=RED,
        )
        grad_glow = grad_arrow.copy().set_color(RED_A).set_opacity(0.35)
        grad_arrow.set_z_index(1)
        grad_glow.set_z_index(1)
        self.add(grad_arrow, grad_glow)

        # Fade out the dot
        self.play(FadeOut(dot), run_time=0.5)

        # --- Rotating direction arrow + slope number ---
        arr_len = 0.25
        # Start further left (CCW) of gradient direction
        start_ang = g_ang + 1.0          # ~57° left of gradient
        angle_tr = ValueTracker(start_ang)

        # Camera orientation for billboard labels
        cam_phi = 60 * DEGREES
        cam_theta = -55 * DEGREES

        def slope_color(norm_slope):
            """RED (positive) → YELLOW (zero) → BLUE (negative)."""
            if norm_slope > 0:
                return interpolate_color(YELLOW, RED, min(norm_slope * 1.2, 1.0))
            else:
                return interpolate_color(YELLOW, BLUE, min(-norm_slope * 1.2, 1.0))

        def build_dir_arrow():
            a = angle_tr.get_value()
            dx, dy = np.cos(a), np.sin(a)
            d_hat = np.array([dx, dy], dtype=float)
            d_hat /= np.linalg.norm(d_hat)
            du, dv = arr_len * d_hat[0], arr_len * d_hat[1]
            s = axes.c2p(u0, v0, z0) + z_bump_grad
            e = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv) + z_bump_grad

            slope = fu * dx + fv * dy
            col = slope_color(slope / grad_mag)

            arr = Arrow3D(
                start=s, end=e,
                thickness=0.010, height=0.09, base_radius=0.03,
                color=col,
            )
            arr.set_z_index(3)
            return arr

        def build_slope_label():
            a = angle_tr.get_value()
            dx, dy = np.cos(a), np.sin(a)
            slope = fu * dx + fv * dy

            sign = "+" if slope >= 0 else ""
            txt = f"{sign}{slope:.2f}"
            label = Text(txt, font_size=12, color=WHITE)

            # Position beyond arrow tip along arrow direction
            d_hat = np.array([dx, dy], dtype=float)
            d_hat /= np.linalg.norm(d_hat)
            overshoot = arr_len + 0.08
            du, dv = overshoot * d_hat[0], overshoot * d_hat[1]
            beyond = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv) + z_bump_grad

            label.move_to(beyond)
            label.rotate(cam_phi, RIGHT)
            label.rotate(cam_theta + PI / 2, OUT)
            label.set_z_index(4)
            return label

        dir_arrow = always_redraw(build_dir_arrow)
        slope_label = always_redraw(build_slope_label)

        p_focus = axes.c2p(u0, v0, z0) + z_bump_grad
        self.play(
            GrowFromPoint(dir_arrow, p_focus),
            FadeIn(slope_label),
            run_time=0.6,
        )

        # --- Phase 1: Align with gradient, then rotate to opposite ---
        self.play(
            angle_tr.animate.set_value(g_ang),
            run_time=2.0,
            rate_func=smooth,
        )
        self.wait(1.0)

        # Rotate: gradient → opposite (slope drops through 0 to most negative)
        self.play(
            angle_tr.animate.set_value(g_ang + PI),
            run_time=6.0,
            rate_func=smooth,
        )

        # "when you move against it, the height drops quickly"
        self.wait(1.0)

        # --- Phase 2: Continue to perpendicular (slope ≈ 0) ---
        perp_ang = g_ang + PI / 2
        self.play(
            angle_tr.animate.set_value(perp_ang),
            run_time=4.0,
            rate_func=smooth,
        )

        # "there's essentially no change at all"
        self.wait(1.2)

        # --- Phase 3: Fade out label only, keep direction arrow, show ∇f ---
        self.play(
            FadeOut(slope_label),
            run_time=0.8,
            rate_func=smooth,
        )

        # ∇f label — right side of screen, shifted left a bit
        nabla_label = MathTex(r"\nabla f", font_size=56, color=RED)
        nabla_label.to_edge(RIGHT, buff=1.0).shift(UP * 0.3)
        nabla_label.set_z_index(10)
        self.add_fixed_in_frame_mobjects(nabla_label)

        self.play(FadeIn(nabla_label, shift=RIGHT * 0.2), run_time=1.0)

        self.wait(2.0)


class S05_5_Transition(ThreeDScene):
    """Transition: strip overlays → bird's-eye → contour-line cross-fade."""
    def construct(self):
        import matplotlib.pyplot as plt

        axes, axes_labels, surface = build_world()

        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)

        # Shift world (matches S04 / S05)
        world_offset = axes.c2p(u0, v0, z0)
        VGroup(axes, surface).shift(-world_offset)

        # Camera — matches S05 end
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-55 * DEGREES,
            zoom=3.2,
            focal_point=ORIGIN,
        )

        surface.set_style(fill_opacity=0.85, stroke_width=0)
        surface.set_z_index(-1)
        self.add(surface)

        # ── Recreate S05 end objects ──────────────────────────
        z_bump_grad = OUT * 0.12
        grad_dir = np.array([fu, fv], dtype=float)
        grad_mag = np.linalg.norm(grad_dir)
        grad_hat = grad_dir / grad_mag
        g_ang = np.arctan2(fv, fu)
        grad_len = 0.25
        gdu, gdv = grad_len * grad_hat[0], grad_len * grad_hat[1]

        grad_start = axes.c2p(u0, v0, z0) + z_bump_grad
        grad_end = axes.c2p(
            u0 + gdu, v0 + gdv,
            z0 + fu * gdu + fv * gdv,
        ) + z_bump_grad

        grad_arrow = Arrow3D(
            start=grad_start, end=grad_end,
            thickness=0.012, height=0.10, base_radius=0.035,
            color=RED,
        )
        grad_glow = grad_arrow.copy().set_color(RED_A).set_opacity(0.35)
        grad_arrow.set_z_index(1)
        grad_glow.set_z_index(1)
        self.add(grad_arrow, grad_glow)

        # Direction arrow at perpendicular (S05 end state)
        arr_len = 0.25
        perp_ang = g_ang + PI / 2
        dx, dy = np.cos(perp_ang), np.sin(perp_ang)
        s_pt = axes.c2p(u0, v0, z0) + z_bump_grad
        e_pt = axes.c2p(
            u0 + arr_len * dx, v0 + arr_len * dy,
            z0 + fu * arr_len * dx + fv * arr_len * dy,
        ) + z_bump_grad

        slope_val = fu * dx + fv * dy

        def _slope_col(ns):
            return (interpolate_color(YELLOW, RED, min(ns * 1.2, 1.0)) if ns > 0
                    else interpolate_color(YELLOW, BLUE, min(-ns * 1.2, 1.0)))

        dir_arrow = Arrow3D(
            start=s_pt, end=e_pt,
            thickness=0.010, height=0.09, base_radius=0.03,
            color=_slope_col(slope_val / grad_mag),
        )
        dir_arrow.set_z_index(3)
        self.add(dir_arrow)

        # ∇f label (fixed in frame)
        nabla_label = MathTex(r"\nabla f", font_size=56, color=RED)
        nabla_label.to_edge(RIGHT, buff=1.0).shift(UP * 0.3)
        nabla_label.set_z_index(10)
        self.add_fixed_in_frame_mobjects(nabla_label)
        self.add(nabla_label)

        self.wait(0.3)

        # ── Phase 1: strip all overlays ──────────────────────
        self.play(
            FadeOut(grad_arrow), FadeOut(grad_glow),
            FadeOut(dir_arrow), FadeOut(nabla_label),
            run_time=1.0,
        )

        # ── Phase 2: pan to centre by sliding the surface (camera fixed) ──
        # Animating objects toward ORIGIN while camera looks at ORIGIN
        # produces a smooth pan effect.
        recentre = -np.array(axes.c2p(0, 0, 0))
        self.play(
            surface.animate.shift(recentre),
            run_time=1.5,
            rate_func=smooth,
        )
        axes.shift(recentre)   # keep in sync (not visible, just for c2p)

        # ── Phase 3: smooth zoom-out → bird's-eye ───────────
        self.move_camera(
            phi=0 * DEGREES,
            theta=-90 * DEGREES,
            zoom=0.85,
            focal_point=ORIGIN,
            run_time=3.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        # ── Phase 3: generate contour curves & cross-fade ───
        N = 300
        us = np.linspace(-3, 3, N)
        vs = np.linspace(-3, 3, N)
        U, V = np.meshgrid(us, vs)
        Z_grid = peaks_f(U, V)

        levels = np.linspace(-5, 8, 14)
        fig, ax = plt.subplots()
        cs = ax.contour(U, V, Z_grid, levels=levels)

        def _z_col(z):
            """Match surface fill_by_value palette."""
            if z <= -8:
                return BLUE_E
            if z <= 0:
                return interpolate_color(BLUE_E, BLUE_C, (z + 8) / 8)
            if z <= 4:
                return interpolate_color(BLUE_C, GREEN_C, z / 4)
            if z <= 8:
                return interpolate_color(GREEN_C, YELLOW_E, (z - 4) / 4)
            return YELLOW_E

        contour_curves = VGroup()
        for i, lev in enumerate(cs.levels):
            segs = cs.allsegs[i]
            col = _z_col(lev)
            for seg in segs:
                if len(seg) < 3:
                    continue
                if len(seg) > 200:
                    idx = np.round(np.linspace(0, len(seg) - 1, 200)).astype(int)
                    seg = seg[idx]
                pts = [axes.c2p(u, v, 0) for u, v in seg]
                curve = VMobject()
                curve.set_points_smoothly(pts)
                curve.set_stroke(col, width=2.0, opacity=0.9)
                curve.set_z_index(5)
                contour_curves.add(curve)

        plt.close(fig)

        # Cross-fade: surface out, contour lines in
        self.play(
            FadeOut(surface),
            FadeIn(contour_curves),
            run_time=2.5,
        )

        self.wait(1.5)


class S06_GradientPerp(ThreeDScene):
    """Level curves → ∇f perpendicular → zero dot product."""
    def construct(self):
        import matplotlib.pyplot as plt

        axes, _, _ = build_world()
        self.set_camera_orientation(
            phi=0 * DEGREES,
            theta=-90 * DEGREES,
            zoom=0.85,
            focal_point=ORIGIN,
        )

        # ── Level curves (matches S05_5_Transition end) ──────────
        N = 300
        Ug, Vg = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-3, 3, N))
        Zg = peaks_f(Ug, Vg)
        levels = np.linspace(-5, 8, 14)

        fig, ax_mpl = plt.subplots()
        cs = ax_mpl.contour(Ug, Vg, Zg, levels=levels)

        def _z_col(z):
            if z <= 0:
                return interpolate_color(BLUE_E, BLUE_C, (z + 8) / 8)
            if z <= 4:
                return interpolate_color(BLUE_C, GREEN_C, z / 4)
            if z <= 8:
                return interpolate_color(GREEN_C, YELLOW_E, (z - 4) / 4)
            return YELLOW_E

        contour_curves = VGroup()
        for i, lev in enumerate(cs.levels):
            col = _z_col(lev)
            for seg in cs.allsegs[i]:
                if len(seg) < 3:
                    continue
                if len(seg) > 200:
                    idx = np.round(np.linspace(0, len(seg) - 1, 200)).astype(int)
                    seg = seg[idx]
                pts = [axes.c2p(u, v, 0) for u, v in seg]
                curve = VMobject()
                curve.set_points_smoothly(pts)
                curve.set_stroke(col, width=2.0, opacity=0.9)
                curve.set_z_index(1)
                contour_curves.add(curve)
        plt.close(fig)

        self.add(contour_curves)
        self.wait(0.5)

        # ── Point: same as all previous scenes ────────────────────
        u0, v0 = 0.9, 1.2
        z0     = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)
        gm       = np.linalg.norm([fu, fv])
        grad_hat = np.array([fu, fv]) / gm
        tang_hat = np.array([-fv, fu]) / gm   # 90° CCW from gradient
        p0 = axes.c2p(u0, v0, 0)
        arr_len = 0.7   # data units

        # ── Phase 1: dim curves, highlight level curve through point ──
        fig2, ax2 = plt.subplots()
        cs2 = ax2.contour(Ug, Vg, Zg, levels=[z0])
        h_curves = VGroup()
        for seg in cs2.allsegs[0]:
            if len(seg) < 3:
                continue
            if len(seg) > 200:
                idx = np.round(np.linspace(0, len(seg) - 1, 200)).astype(int)
                seg = seg[idx]
            pts = [axes.c2p(u, v, 0) for u, v in seg]
            c = VMobject()
            c.set_points_smoothly(pts)
            c.set_stroke(WHITE, width=4.0, opacity=1.0)
            c.set_z_index(6)
            h_curves.add(c)
        plt.close(fig2)

        dot = Dot3D(p0, radius=0.06, color=WHITE)
        dot.set_z_index(10)

        self.play(
            contour_curves.animate.set_stroke(opacity=0.2),
            FadeIn(h_curves),
            run_time=1.0,
        )
        self.play(FadeIn(dot, scale=2.5), run_time=0.6)
        self.wait(0.3)

        # ── Phase 2: tangent arrow (along level curve = same height) ──
        tang_end = axes.c2p(
            u0 + arr_len * tang_hat[0],
            v0 + arr_len * tang_hat[1],
            0,
        )
        tang_arrow = Arrow(
            p0, tang_end, buff=0, color=BLUE_C,
            stroke_width=5, max_tip_length_to_length_ratio=0.2,
        )
        tang_arrow.set_z_index(8)
        self.play(GrowArrow(tang_arrow), run_time=0.8)
        self.wait(0.4)

        # ── Phase 3: gradient arrow + right-angle marker ──────────
        grad_end = axes.c2p(
            u0 + arr_len * grad_hat[0],
            v0 + arr_len * grad_hat[1],
            0,
        )
        grad_arrow_obj = Arrow(
            p0, grad_end, buff=0, color=RED,
            stroke_width=5, max_tip_length_to_length_ratio=0.2,
        )
        grad_arrow_obj.set_z_index(8)

        nabla_lbl = MathTex(r"\nabla f", color=RED, font_size=44)
        nabla_lbl.to_edge(RIGHT, buff=0.6).shift(UP * 0.8)
        self.add_fixed_in_frame_mobjects(nabla_lbl)

        # right-angle box in scene space
        sq = 0.12
        rg  = axes.c2p(u0 + sq * grad_hat[0], v0 + sq * grad_hat[1], 0)
        rt  = axes.c2p(u0 + sq * tang_hat[0], v0 + sq * tang_hat[1], 0)
        rgt = axes.c2p(
            u0 + sq * grad_hat[0] + sq * tang_hat[0],
            v0 + sq * grad_hat[1] + sq * tang_hat[1],
            0,
        )
        ra = VGroup(
            Line(rg, rgt, color=WHITE, stroke_width=2),
            Line(rt, rgt, color=WHITE, stroke_width=2),
        ).set_z_index(9)

        self.play(
            GrowArrow(grad_arrow_obj),
            FadeIn(nabla_lbl, shift=RIGHT * 0.1),
            run_time=0.8,
        )
        self.play(Create(ra), run_time=0.5)
        self.wait(0.8)

        # ── Phase 4: "zero overlap" — dot product animation ───────
        # Fade out blue tangent arrow; replace with yellow test direction
        # that starts 25° from ∇f (large overlap) and rotates to 90° (zero).
        self.play(FadeOut(tang_arrow), run_time=0.4)

        angle_tr = ValueTracker(25 * DEGREES)

        def build_test_arr():
            θ = angle_tr.get_value()
            d = np.cos(θ) * grad_hat + np.sin(θ) * tang_hat
            end_uv = np.array([u0, v0]) + arr_len * d
            a = Arrow(
                p0, axes.c2p(end_uv[0], end_uv[1], 0),
                buff=0, color=YELLOW,
                stroke_width=5, max_tip_length_to_length_ratio=0.2,
            )
            a.set_z_index(9)
            return a

        def build_proj():
            """Yellow bar along ∇f showing projection (overlap)."""
            θ = angle_tr.get_value()
            proj = arr_len * np.cos(θ)
            bar = VMobject()
            if proj > 0.01:
                end_uv = np.array([u0, v0]) + proj * grad_hat
                bar = Line(
                    p0, axes.c2p(end_uv[0], end_uv[1], 0),
                    color=YELLOW, stroke_width=10,
                )
                bar.set_z_index(7)
            return bar

        test_arr = always_redraw(build_test_arr)
        proj_bar = always_redraw(build_proj)

        overlap_lbl = Text("overlap", font_size=26, color=YELLOW)
        overlap_lbl.move_to(LEFT * 2.0 + UP * 1.2)
        self.add_fixed_in_frame_mobjects(overlap_lbl)

        self.play(
            FadeIn(test_arr),
            FadeIn(proj_bar),
            FadeIn(overlap_lbl),
            run_time=0.8,
        )
        self.wait(0.5)

        # Rotate direction to perpendicular → overlap bar shrinks to zero
        self.play(
            angle_tr.animate.set_value(PI / 2),
            run_time=2.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        # ∇f · d = 0 label
        self.play(FadeOut(overlap_lbl), run_time=0.3)
        formula = MathTex(r"\nabla f \cdot \mathbf{d} = 0", color=WHITE, font_size=48)
        formula.to_edge(DOWN, buff=1.8)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(FadeIn(formula, shift=UP * 0.15), run_time=1.0)

        self.wait(2.0)


class S07_Applications(ThreeDScene):
    """Physics → Gradient Descent → SGD — smooth gradient vignettes."""
    def construct(self):
        import matplotlib.pyplot as plt

        self.set_camera_orientation(
            phi=0 * DEGREES,
            theta=-90 * DEGREES,
            zoom=0.85,
        )

        axes, _, _ = build_world()
        N = 250
        Us, Vs = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-3, 3, N))

        # ── helper: colored VGroup for Transform morph ───────────────────
        # Each level is a separate VMobject (color sticks to submobject).
        # All groups use N_MORPH levels so Transform maps 1-to-1 between panels.
        N_MORPH = 12

        def make_morph(Z_data, levels, colors, lw=2.5):
            """Returns VGroup of N_MORPH VMobjects, one per contour level.
            Colours are per-level; same count across all panels enables clean
            1-to-1 Transform morphing."""
            fig, am = plt.subplots()
            cs = am.contour(Us, Vs, Z_data, levels=levels)
            n_lev  = len(cs.levels)
            n_col  = len(colors) - 1
            level_mobs = []
            for i in range(n_lev):
                t   = i / max(n_lev - 1, 1)
                lo  = int(t * n_col)
                hi  = min(lo + 1, n_col)
                col = interpolate_color(colors[lo], colors[hi], t * n_col - lo)
                lev_mob  = VMobject()
                has_segs = False
                for seg in cs.allsegs[i]:
                    if len(seg) < 3:
                        continue
                    if len(seg) > 80:
                        idx = np.round(np.linspace(0, len(seg) - 1, 80)).astype(int)
                        seg = seg[idx]
                    sub = VMobject()
                    sub.set_points_smoothly([axes.c2p(u, v, 0) for u, v in seg])
                    lev_mob.append_vectorized_mobject(sub)
                    has_segs = True
                if has_segs:
                    lev_mob.set_stroke(col, width=lw, opacity=0.9)
                else:
                    # invisible dummy so the level count stays fixed
                    lev_mob.set_points_smoothly([axes.c2p(0, 0, 0)] * 4)
                    lev_mob.set_stroke(opacity=0)
                level_mobs.append(lev_mob)
            plt.close(fig)
            return VGroup(*level_mobs).set_z_index(2), cs

        def temp_f(u, v):
            return 3.0 * np.exp(-(u ** 2 + v ** 2) / 1.5)

        def loss_f(u, v):
            return 0.12 * u ** 2 + 6.0 * v ** 2

        # ── build all morph objects upfront (same N_MORPH levels each) ───
        Zp = peaks_f(Us, Vs)
        Zt = temp_f(Us, Vs)
        Zl = loss_f(Us, Vs)

        # peaks colours: cold-deep → warm-high (matching scene S01–S06 palette)
        PKS_COLS = [BLUE_E, BLUE_C, GREEN_C, YELLOW_C, YELLOW_E]
        # temperature: outer=cold=blue → inner=hot=red
        TMP_COLS = [BLUE_B, TEAL_C, YELLOW_C, ORANGE, RED]
        # loss bowl: inner=low=blue → outer=high=warm
        LOS_COLS = [BLUE_E, BLUE_C, TEAL_C, GREEN_C, YELLOW_C, YELLOW]

        cur_curves, _ = make_morph(Zp, np.linspace(-5, 8, N_MORPH),    PKS_COLS)
        cur_curves.set_stroke(opacity=0.2)   # S06 dimmed state

        tc_tgt, cs_tc = make_morph(Zt, np.linspace(0.15, 2.7,  N_MORPH), TMP_COLS)
        pk_tgt, _     = make_morph(Zp, np.linspace(-5, 8,       N_MORPH), PKS_COLS)
        lc_tgt, _     = make_morph(Zl, np.linspace(0.15, 11,    N_MORPH), LOS_COLS)

        # ── S06 end state ─────────────────────────────────────────
        u0, v0 = 0.9, 1.2
        z0     = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)
        gm     = np.linalg.norm([fu, fv])
        gh     = np.array([fu, fv]) / gm
        th     = np.array([-fv, fu]) / gm
        p0     = axes.c2p(u0, v0, 0)
        arl    = 0.7

        fig_h, am_h = plt.subplots()
        cs_h = am_h.contour(Us, Vs, Zp, levels=[z0])
        hc = VGroup()
        for seg in cs_h.allsegs[0]:
            if len(seg) < 3:
                continue
            if len(seg) > 200:
                idx = np.round(np.linspace(0, len(seg) - 1, 200)).astype(int)
                seg = seg[idx]
            c = VMobject().set_points_smoothly([axes.c2p(u, v, 0) for u, v in seg])
            c.set_stroke(WHITE, width=4, opacity=1.0).set_z_index(6)
            hc.add(c)
        plt.close(fig_h)

        dot  = Dot3D(p0, radius=0.06, color=WHITE).set_z_index(10)
        gobj = Arrow(p0, axes.c2p(u0 + arl * gh[0], v0 + arl * gh[1], 0),
                     buff=0, color=RED, stroke_width=5,
                     max_tip_length_to_length_ratio=0.2).set_z_index(8)
        perp_ang = np.arctan2(fv, fu) + PI / 2
        pdx, pdy = np.cos(perp_ang), np.sin(perp_ang)
        tobj = Arrow(p0, axes.c2p(u0 + arl * pdx, v0 + arl * pdy, 0),
                     buff=0, color=YELLOW, stroke_width=5,
                     max_tip_length_to_length_ratio=0.2).set_z_index(9)
        sq = 0.12
        ra = VGroup(
            Line(axes.c2p(u0 + sq * gh[0], v0 + sq * gh[1], 0),
                 axes.c2p(u0 + sq * (gh[0] + th[0]), v0 + sq * (gh[1] + th[1]), 0),
                 color=WHITE, stroke_width=2),
            Line(axes.c2p(u0 + sq * th[0], v0 + sq * th[1], 0),
                 axes.c2p(u0 + sq * (gh[0] + th[0]), v0 + sq * (gh[1] + th[1]), 0),
                 color=WHITE, stroke_width=2),
        ).set_z_index(9)

        nabla_lbl = MathTex(r"\nabla f", color=RED, font_size=44)
        nabla_lbl.to_edge(RIGHT, buff=0.6).shift(UP * 0.8)
        s06_fml   = MathTex(r"\nabla f \cdot \mathbf{d} = 0", color=WHITE, font_size=48)
        s06_fml.to_edge(DOWN, buff=1.8)
        self.add_fixed_in_frame_mobjects(nabla_lbl, s06_fml)
        self.add(cur_curves, hc, dot, gobj, tobj, ra)
        self.wait(0.3)

        # ══════════════════════════════════════════════════════════
        # PANEL 1 — Physics: Heat Conduction  J = −k∇T
        # ══════════════════════════════════════════════════════════
        phys_ttl = Text("Physics", font_size=36, color=WHITE, weight=BOLD)
        phys_ttl.to_edge(UP, buff=0.3)
        phys_fml = MathTex(r"\mathbf{J} = -k\,\nabla T", color=BLUE_B, font_size=32)
        phys_fml.next_to(phys_ttl, DOWN, buff=0.18)

        # 1a. Fade out ALL S06 overlays (elements first, THEN morph)
        self.play(
            LaggedStart(
                FadeOut(ra), FadeOut(tobj), FadeOut(gobj), FadeOut(dot), FadeOut(hc),
                lag_ratio=0.15,
            ),
            FadeOut(nabla_lbl, shift=UP * 0.15),
            FadeOut(s06_fml,   shift=DOWN * 0.15),
            run_time=0.9, rate_func=smooth,
        )
        # 1b. Morph curves peaks → isotherms
        self.play(Transform(cur_curves, tc_tgt), run_time=1.4, rate_func=smooth)
        # 1c. Title only (formula NOT in scene yet)
        self.add_fixed_in_frame_mobjects(phys_ttl)
        self.play(Write(phys_ttl), run_time=0.7)

        # Heat-source glow
        src = axes.c2p(0.0, 0.0, 0)
        flame = VGroup(
            Circle(radius=0.38).set_fill(RED, 0.07).set_stroke(width=0),
            Circle(radius=0.22).set_fill(ORANGE, 0.25).set_stroke(width=0),
            Circle(radius=0.10).set_fill(YELLOW, 0.80).set_stroke(width=0),
            Dot(radius=0.045, color=WHITE),
        ).move_to(src).set_z_index(8)
        self.play(GrowFromCenter(flame), run_time=0.5)

        # Scattered arrows on curves, min-separation enforced
        np.random.seed(42)
        all_cands = []
        for i in range(len(cs_tc.levels)):
            for seg in cs_tc.allsegs[i]:
                if len(seg) < 6:
                    continue
                n_sample = max(3, len(seg) // 5)
                idxs = np.round(np.linspace(0, len(seg) - 1, n_sample)).astype(int)
                for idx in idxs:
                    all_cands.append(tuple(seg[idx]))
        np.random.shuffle(all_cands)

        pos_arrs, neg_arrs, placed_uv = [], [], []
        min_sep = 0.60
        for (ui, vi) in all_cands:
            if any((ui - cu) ** 2 + (vi - cv) ** 2 < min_sep ** 2 for cu, cv in placed_uv):
                continue
            tu = (temp_f(ui + 0.01, vi) - temp_f(ui - 0.01, vi)) / 0.02
            tv = (temp_f(ui, vi + 0.01) - temp_f(ui, vi - 0.01)) / 0.02
            mag = np.sqrt(tu ** 2 + tv ** 2)
            if mag < 0.03:
                continue
            L  = 0.25
            sp = axes.c2p(ui, vi, 0)
            pos_arrs.append(Arrow(
                sp, axes.c2p(ui + tu / mag * L, vi + tv / mag * L, 0),
                buff=0, color=RED, stroke_width=3,
                max_tip_length_to_length_ratio=0.28,
            ).set_z_index(5))
            neg_arrs.append(Arrow(
                sp, axes.c2p(ui - tu / mag * L, vi - tv / mag * L, 0),
                buff=0, color=BLUE_B, stroke_width=3,
                max_tip_length_to_length_ratio=0.28,
            ).set_z_index(5))
            placed_uv.append((ui, vi))

        self.play(
            LaggedStart(*[GrowArrow(a) for a in pos_arrs], lag_ratio=0.07),
            run_time=1.5,
        )
        # 1d. Formula only after arrows are up (added to fixed-frame here for first time)
        self.add_fixed_in_frame_mobjects(phys_fml)
        self.play(Write(phys_fml), run_time=0.8)
        self.wait(0.3)
        self.play(
            *[Transform(pa, na) for pa, na in zip(pos_arrs, neg_arrs)],
            run_time=0.9,
        )
        self.wait(1.2)

        # ══════════════════════════════════════════════════════════
        # PANEL 2 — Numerical Methods: Gradient Descent
        # ══════════════════════════════════════════════════════════
        num_ttl = Text("Numerical Methods", font_size=36, color=WHITE, weight=BOLD)
        num_ttl.to_edge(UP, buff=0.3)
        num_fml = MathTex(
            r"\mathbf{x}_{n+1} = \mathbf{x}_n - \alpha\,\nabla f(\mathbf{x}_n)",
            color=YELLOW, font_size=28,
        )
        num_fml.next_to(num_ttl, DOWN, buff=0.18)

        # 2a. Fade out Physics elements first
        self.play(
            FadeOut(phys_ttl, shift=UP * 0.2),
            FadeOut(phys_fml, shift=UP * 0.2),
            FadeOut(flame),
            LaggedStart(*[FadeOut(a) for a in pos_arrs], lag_ratio=0.05),
            run_time=0.8, rate_func=smooth,
        )
        # 2b. Morph curves isotherms → peaks
        self.play(Transform(cur_curves, pk_tgt), run_time=1.4, rate_func=smooth)
        # 2c. Title only
        self.add_fixed_in_frame_mobjects(num_ttl)
        self.play(Write(num_ttl), run_time=0.7)

        # Gradient descent path
        lr, ux, vx = 0.10, -2.0, 1.6
        gd_pts = [axes.c2p(ux, vx, 0)]
        for _ in range(14):
            gfu_v, gfv_v = grad_numeric(peaks_f, ux, vx)
            ux = np.clip(ux - lr * gfu_v, -2.9, 2.9)
            vx = np.clip(vx - lr * gfv_v, -2.9, 2.9)
            gd_pts.append(axes.c2p(ux, vx, 0))

        gd_line  = VMobject().set_points_smoothly(gd_pts)
        gd_line.set_stroke(YELLOW, width=4).set_z_index(6)
        gd_hops  = VGroup(*[Dot3D(p, radius=0.045, color=YELLOW).set_z_index(7) for p in gd_pts])
        gd_start = Dot3D(gd_pts[0],  radius=0.08, color=YELLOW).set_z_index(9)
        gd_end   = Dot3D(gd_pts[-1], radius=0.09, color=GREEN_C).set_z_index(9)

        self.play(FadeIn(gd_start, scale=2.0), run_time=0.4)
        self.play(Create(gd_line), FadeIn(gd_hops), run_time=2.0, rate_func=smooth)
        self.play(FadeIn(gd_end, scale=2.5), run_time=0.4)
        # 2d. Formula after path is fully drawn
        self.add_fixed_in_frame_mobjects(num_fml)
        self.play(Write(num_fml), run_time=0.8)
        self.wait(1.0)

        # ══════════════════════════════════════════════════════════
        # PANEL 3 — Machine Learning: SGD
        # ══════════════════════════════════════════════════════════
        ml_ttl  = Text("Machine Learning", font_size=36, color=WHITE, weight=BOLD)
        ml_ttl.to_edge(UP, buff=0.3)
        sgd_fml = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} - \alpha\,\nabla_{\!\mathcal{B}}\,\mathcal{L}(\mathbf{w})",
            color=YELLOW, font_size=28,
        )
        sgd_fml.next_to(ml_ttl, DOWN, buff=0.18)

        # 3a. Fade out Numerical elements first
        self.play(
            FadeOut(num_ttl, shift=UP * 0.2),
            FadeOut(num_fml, shift=UP * 0.2),
            FadeOut(gd_line), FadeOut(gd_hops), FadeOut(gd_start), FadeOut(gd_end),
            run_time=0.8, rate_func=smooth,
        )
        # 3b. Morph curves peaks → loss ellipses
        self.play(Transform(cur_curves, lc_tgt), run_time=1.4, rate_func=smooth)
        # 3c. Title only
        self.add_fixed_in_frame_mobjects(ml_ttl)
        self.play(Write(ml_ttl), run_time=0.7)

        # SGD path
        np.random.seed(7)
        lr2, w1, w2 = 0.16, 2.5, 0.55
        sgd_pts = [axes.c2p(w1, w2, 0)]
        for _ in range(28):
            gw1 = 2 * 0.12 * w1 + np.random.normal(0, 0.04)
            gw2 = 2 * 6.0  * w2 + np.random.normal(0, 0.04)
            w1  = np.clip(w1 - lr2 * gw1, -2.9, 2.9)
            w2  = np.clip(w2 - lr2 * gw2, -2.9, 2.9)
            sgd_pts.append(axes.c2p(w1, w2, 0))

        sgd_line  = VMobject().set_points_smoothly(sgd_pts)
        sgd_line.set_stroke(YELLOW, width=4).set_z_index(6)
        sgd_hops  = VGroup(*[Dot3D(p, radius=0.04, color=YELLOW).set_z_index(7) for p in sgd_pts])
        sgd_start = Dot3D(sgd_pts[0],  radius=0.08, color=YELLOW).set_z_index(9)
        sgd_end   = Dot3D(sgd_pts[-1], radius=0.09, color=GREEN_C).set_z_index(9)

        self.play(FadeIn(sgd_start, scale=2.0), run_time=0.4)
        self.play(Create(sgd_line), FadeIn(sgd_hops), run_time=2.5, rate_func=smooth)
        self.play(FadeIn(sgd_end, scale=2.5), run_time=0.4)
        # 3d. Formula after path is fully drawn
        self.add_fixed_in_frame_mobjects(sgd_fml)
        self.play(Write(sgd_fml), run_time=0.8)
        self.wait(2.0)
