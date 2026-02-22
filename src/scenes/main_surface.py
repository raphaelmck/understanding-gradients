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

class S03_InfiniteSlopes(ThreeDScene):
    def construct(self):
        # -----------------------------------------------------
        # 1. Setup & Camera Sync
        # -----------------------------------------------------
        axes, axes_labels, surface = build_world()

        u0, v0 = 0.9, 1.2
        z0 = peaks_f(u0, v0)
        fu, fv = grad_numeric(peaks_f, u0, v0)

        # Shift the ENTIRE world so the point of interest sits at the 3D origin.
        # This guarantees the point is dead-center regardless of camera angles.
        world_offset = axes.c2p(u0, v0, z0)
        world = VGroup(axes, surface)
        world.shift(-world_offset)

        # p_focus is now at ORIGIN
        p_focus = ORIGIN

        self.set_camera_orientation(
            phi=55 * DEGREES,
            theta=-15 * DEGREES,
            zoom=1.9,
            focal_point=ORIGIN,
        )

        surface.set_style(fill_opacity=0.85, stroke_width=0)
        dot = Dot3D(ORIGIN, radius=0.05, color=WHITE)
        self.add(surface, dot)

        # -----------------------------------------------------
        # 2. Smooth Centered Zoom
        # -----------------------------------------------------
        self.move_camera(
            phi=70 * DEGREES,
            theta=-10 * DEGREES,
            zoom=3.0,
            focal_point=ORIGIN,
            run_time=2.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        # -----------------------------------------------------
        # 3. Infinite Slopes (Clockwise from Downhill)
        # -----------------------------------------------------
        max_slope = np.linalg.norm([fu, fv])
        g_ang = np.arctan2(fv, fu)

        num_arrows = 12
        start_angle = g_ang + PI
        angles_to_show = [start_angle - i * (TAU / num_arrows) for i in range(num_arrows)]

        ghost_arrows = VGroup()

        for i, angle in enumerate(angles_to_show):
            dx, dy = np.cos(angle), np.sin(angle)

            slope = fu * dx + fv * dy
            norm_slope = slope / max_slope

            # Vivid, saturated colors
            if norm_slope > 0:
                col = interpolate_color(YELLOW, RED, min(norm_slope * 1.2, 1.0))
            else:
                col = interpolate_color(TEAL, BLUE, min(-norm_slope * 1.2, 1.0))

            # --- Vector setup ---
            d = np.array([dx, dy], dtype=float)
            d /= np.linalg.norm(d)
            arr_len = 0.25
            du, dv = arr_len * d[0], arr_len * d[1]
            start_pt = axes.c2p(u0, v0, z0)
            end_pt = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv)

            arr = Arrow3D(
                start=start_pt,
                end=end_pt,
                thickness=0.004,
                height=0.07,
                base_radius=0.02,
                color=col,
            )

            # Floating Value Label
            sign = "+" if slope > 0.05 else ("" if slope > -0.05 else "")
            val_text = f"{sign}{slope:.2f}"
            val_label = Text(val_text, font_size=16, color=col)
            val_label.move_to(arr.get_end() + UP * 0.1)
            val_label.rotate(PI / 2, RIGHT)
            val_label.rotate(-10 * DEGREES, UP)

            # --- Particle trail ---
            trail_head = ValueTracker(0.001)
            trail_len = 0.25  # constant streak length

            def make_particle_trail(dx_v, dy_v, head_tr, streak_len):
                def _build():
                    h = head_tr.get_value()
                    t = max(0.0, h - streak_len)
                    if h < 0.005:
                        return VGroup()
                    opac = max(0.0, 1.0 - h / 1.5)
                    if opac < 0.01:
                        return VGroup()
                    trail = ParametricFunction(
                        lambda s: axes.c2p(
                            u0 + s * dx_v,
                            v0 + s * dy_v,
                            peaks_f(u0 + s * dx_v, v0 + s * dy_v),
                        ),
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

            snake = always_redraw(make_particle_trail(dx, dy, trail_head, trail_len))
            ghost_arrows.add(arr)

            # Phase 1: Arrow appears with label
            self.play(
                GrowFromPoint(arr, p_focus),
                FadeIn(val_label, shift=UP * 0.05),
                run_time=0.5,
                rate_func=smooth,
            )

            # Phase 2: Particle trail ripples out
            self.add(snake)
            self.play(
                trail_head.animate.set_value(1.5),
                run_time=1.3,
                rate_func=linear,
            )
            self.remove(snake)

            # Phase 3: Fade label, dim arrow to low opacity (keep its color)
            self.play(
                FadeOut(val_label, shift=UP * 0.05),
                arr.animate.set_opacity(0.15),
                run_time=0.25,
                rate_func=smooth,
            )

            # Pause between arrows
            if i < num_arrows - 1:
                self.wait(0.1)

        self.wait(2.0)
