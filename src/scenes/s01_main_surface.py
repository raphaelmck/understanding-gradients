from manim import *
import numpy as np

# Manim Community Edition

class GradientIntro(ThreeDScene):
    def construct(self):
        # --- Camera ---
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # --- Axes ---
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

        # --- Function definition (peaks) ---
        def f(u, v):
            return (
                3 * (1 - u) ** 2 * np.exp(-u**2 - (v + 1) ** 2)
                - 10 * (u / 5 - u**3 - v**5) * np.exp(-u**2 - v**2)
                - (1 / 3) * np.exp(-(u + 1) ** 2 - v**2)
            )

        def peaks_point(u, v):
            return axes.c2p(u, v, f(u, v))

        # --- Surface ---
        surface = Surface(
            lambda u, v: peaks_point(u, v),
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

        # --- 0) Cold open: surface + gentle spin (no axes yet) ---
        self.play(Create(surface), run_time=2.8)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(2.0)

        # --- 1) Stop spin, reveal axes ---
        self.stop_ambient_camera_rotation()
        self.play(FadeIn(axes), FadeIn(axes_labels), run_time=1.2)

        # --- 2) Choose a point + height line ---
        u0, v0 = 0.8, -0.3
        z0 = f(u0, v0)

        p_ground = axes.c2p(u0, v0, 0)
        p_surface = axes.c2p(u0, v0, z0)

        ground_dot = Dot3D(p_ground, radius=0.06, color=WHITE)
        surf_dot = Dot3D(p_surface, radius=0.07, color=WHITE)

        height_line = DashedLine(p_ground, p_surface, dash_length=0.08).set_color(GRAY_B)

        # Value label near the surface dot
        val_label = DecimalNumber(z0, num_decimal_places=2)
        val_label.scale(0.5).set_color(WHITE)
        val_label.next_to(surf_dot, UR, buff=0.15)

        # Keep label facing camera-ish (good enough for this use)
        val_label.add_updater(lambda m: m.become(
            DecimalNumber(f(u0, v0), num_decimal_places=2).scale(0.5).set_color(WHITE).next_to(surf_dot, UR, buff=0.15)
        ))

        self.play(FadeIn(ground_dot), FadeIn(surf_dot), Create(height_line), FadeIn(val_label), run_time=1.2)
        self.wait(0.4)

        # --- 3) Tangent plane patch (local linearization) ---
        def grad_numeric(u, v, eps=1e-3):
            fu = (f(u + eps, v) - f(u - eps, v)) / (2 * eps)
            fv = (f(u, v + eps) - f(u, v - eps)) / (2 * eps)
            return fu, fv

        fu, fv = grad_numeric(u0, v0)

        # small tangent plane: z = z0 + fu*(x-u0) + fv*(y-v0)
        plane = Surface(
            lambda a, b: axes.c2p(a, b, z0 + fu * (a - u0) + fv * (b - v0)),
            u_range=[u0 - 0.8, u0 + 0.8],
            v_range=[v0 - 0.8, v0 + 0.8],
            resolution=(10, 10),
        )
        plane.set_style(fill_opacity=0.25, fill_color=WHITE, stroke_width=0)

        self.play(FadeIn(plane), run_time=0.8)

        # --- 4) A few directional arrows on the tangent plane + sample numbers ---
        # helper: arrow in direction (dx,dy) from (u0,v0) on the tangent plane
        def tangent_arrow(dx, dy, length=0.9, color=BLUE_A):
            d = np.array([dx, dy], dtype=float)
            d = d / np.linalg.norm(d)
            du, dv = length * d[0], length * d[1]
            start = axes.c2p(u0, v0, z0)
            end = axes.c2p(u0 + du, v0 + dv, z0 + fu * du + fv * dv)
            return Arrow3D(start, end, thickness=0.02, color=color)

        # directional derivative (approx): fu*vx + fv*vy for unit (vx,vy)
        def dir_slope(dx, dy):
            d = np.array([dx, dy], dtype=float)
            d = d / np.linalg.norm(d)
            return fu * d[0] + fv * d[1]

        dirs = [(1, 0), (0.4, 1), (-1, 0.2), (0.2, -1)]
        arrows = VGroup()
        labels = VGroup()

        for (dx, dy) in dirs:
            arr = tangent_arrow(dx, dy, color=BLUE_B)
            s = dir_slope(dx, dy)
            lab = DecimalNumber(s, num_decimal_places=2).scale(0.35).set_color(WHITE)
            # place label near arrow end
            lab.move_to(arr.get_end() + 0.18 * UP)
            arrows.add(arr)
            labels.add(lab)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), run_time=1.2)
        self.play(FadeIn(labels), run_time=0.6)
        self.wait(0.6)

        # --- 5) Fade to the special arrow: gradient direction ---
        gxy = np.array([fu, fv], dtype=float)
        gxy_norm = np.linalg.norm(gxy)
        if gxy_norm < 1e-9:
            g_dir = np.array([1.0, 0.0])
        else:
            g_dir = gxy / gxy_norm

        grad_arrow = tangent_arrow(g_dir[0], g_dir[1], length=1.2, color=YELLOW)
        grad_label = MathTex(r"\nabla f").scale(0.6).set_color(YELLOW)
        grad_label.move_to(grad_arrow.get_end() + 0.25 * UP)

        self.play(
            FadeOut(arrows),
            FadeOut(labels),
            run_time=0.7
        )
        self.play(GrowArrow(grad_arrow), FadeIn(grad_label), run_time=0.9)
        self.wait(0.4)

        # --- 6) Rotating direction arrow + live "overlap" number (fixed in frame) ---
        theta = ValueTracker(0.0)

        def rotating_arrow():
            ang = theta.get_value()
            d = np.array([np.cos(ang), np.sin(ang)])
            return tangent_arrow(d[0], d[1], length=1.05, color=BLUE_A)

        rot_arr = always_redraw(rotating_arrow)

        # live directional derivative value = fu*cos + fv*sin
        value = DecimalNumber(0.0, num_decimal_places=2).scale(0.55).set_color(WHITE)
        value_label = MathTex(r"D_{\mathbf{v}}f \;\approx\; \nabla f \cdot \mathbf{v}").scale(0.55).set_color(GRAY_A)

        hud = VGroup(value_label, value).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        hud.to_corner(UL, buff=0.4)
        self.add_fixed_in_frame_mobjects(hud)

        def update_value(m):
            ang = theta.get_value()
            m.set_value(fu * np.cos(ang) + fv * np.sin(ang))
            return m

        value.add_updater(update_value)

        self.play(FadeIn(hud), FadeIn(rot_arr), run_time=0.7)

        # Rotate through directions
        self.play(theta.animate.set_value(TAU), run_time=4.0, rate_func=linear)
        self.wait(0.3)

        # Show "perpendicular ~ 0": rotate to perpendicular of gradient direction
        g_ang = np.arctan2(g_dir[1], g_dir[0])
        self.play(theta.animate.set_value(g_ang + PI / 2), run_time=1.2)
        self.wait(0.5)

        # Align with gradient to show maximum
        self.play(theta.animate.set_value(g_ang), run_time=1.2)
        self.wait(0.8)

        # clean exit hold
        self.wait(1.0)
