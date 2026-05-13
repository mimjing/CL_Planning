import numpy as np
from typing import Optional, Union

class HeadTopDownRenderer:
    """
    This class is used to render the top-down view of the environment.
    It is a pseudo-render function, only used to update onscreen message when using panda3d backend.
    """

    def __init__(self, env):
        self.engine = env.engine
        self.head_top_down_renderer = None
        self.reference_lines = None
        self.diagnostic_ego_pose = None
    
    def set_reference_lines(self, ref_lines):
        """Set reference lines to be drawn on the next render call."""
        self.reference_lines = ref_lines
    
    def set_diagnostic_state(self, ego_position, ego_heading):
        """Set ego pose for diagnostic arrows."""
        def _to_numpy(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            elif hasattr(x, "cpu"):
                return x.cpu().numpy()
            return np.asarray(x)
            
        self.diagnostic_ego_pose = (_to_numpy(ego_position), _to_numpy(ego_heading))
    
    def _draw_arrow_on_canvas(self, canvas, x, y, heading, color, length=6.0, width=6, head_width=1.5):
        import math
        import pygame

        if hasattr(x, 'item'): x = x.item()
        if hasattr(y, 'item'): y = y.item()
        if hasattr(heading, 'item'): heading = heading.item()

        dx = math.cos(heading) * length
        dy = math.sin(heading) * length
        start_pix = canvas.pos2pix(x, y)
        end_pix = canvas.pos2pix(x + dx, y + dy)

        pygame.draw.line(canvas, color, start_pix, end_pix, width)

        perp_heading = heading + math.pi / 2
        px1 = x + dx - math.cos(perp_heading) * head_width
        py1 = y + dy - math.sin(perp_heading) * head_width
        px2 = x + dx + math.cos(perp_heading) * head_width
        py2 = y + dy + math.sin(perp_heading) * head_width

        p1_pix = canvas.pos2pix(px1, py1)
        p2_pix = canvas.pos2pix(px2, py2)
        pygame.draw.polygon(canvas, color, [end_pix, p1_pix, p2_pix])

    def _get_nearest_centerline_tangent(self, ego_xy):
        if self.reference_lines is None:
            return None

        best = None
        best_dist = float("inf")

        if isinstance(self.reference_lines, dict):
            positions = self.reference_lines.get("position", [])
            orientations = self.reference_lines.get("orientation", None)
            valid_masks = self.reference_lines.get("valid_mask", None)
            num_lines = len(positions)

            for i in range(num_lines):
                pos = np.asarray(positions[i])
                if pos.size == 0:
                    continue
                pos_xy = pos[..., :2]

                mask = None
                if valid_masks is not None and len(valid_masks) > i:
                    mask = np.asarray(valid_masks[i]).astype(bool)
                    if mask.shape[0] == pos_xy.shape[0]:
                        pos_xy = pos_xy[mask]
                    else:
                        mask = None

                if pos_xy.shape[0] == 0:
                    continue

                dists = np.sum((pos_xy - ego_xy) ** 2, axis=1)
                idx = int(np.argmin(dists))

                heading = None
                if orientations is not None and len(orientations) > i:
                    ori = np.asarray(orientations[i])
                    if mask is not None and ori.shape[0] == mask.shape[0]:
                        ori = ori[mask]
                    if ori.shape[0] > idx:
                        heading = float(ori[idx])

                if heading is None:
                    prev_idx = max(idx - 1, 0)
                    next_idx = min(idx + 1, pos_xy.shape[0] - 1)
                    vec = pos_xy[next_idx] - pos_xy[prev_idx]
                    if np.linalg.norm(vec) < 1e-6:
                        heading = 0.0
                    else:
                        heading = float(np.arctan2(vec[1], vec[0]))

                if dists[idx] < best_dist:
                    best_dist = dists[idx]
                    best = (pos_xy[idx], heading)
        else:
            for ref_line in self.reference_lines:
                if len(ref_line) == 0:
                    continue
                line = np.asarray(ref_line)
                pos_xy = line[:, :2]
                dists = np.sum((pos_xy - ego_xy) ** 2, axis=1)
                idx = int(np.argmin(dists))
                if line.shape[1] > 2:
                    heading = float(line[idx, 2])
                else:
                    prev_idx = max(idx - 1, 0)
                    next_idx = min(idx + 1, pos_xy.shape[0] - 1)
                    vec = pos_xy[next_idx] - pos_xy[prev_idx]
                    heading = float(np.arctan2(vec[1], vec[0])) if np.linalg.norm(vec) > 1e-6 else 0.0

                if dists[idx] < best_dist:
                    best_dist = dists[idx]
                    best = (pos_xy[idx], heading)

        return best

    def _draw_diagnostics_on_canvas(self, canvas):
        if self.diagnostic_ego_pose is None:
            return

        ego_position, ego_heading = self.diagnostic_ego_pose
        ego_xy = np.asarray(ego_position, dtype=np.float64)[:2]

        # Ego heading arrow
        self._draw_arrow_on_canvas(canvas, ego_xy[0], ego_xy[1], ego_heading, color=(0, 255, 0), length=15.0, width=5)

        # Nearest centerline tangent arrow
        nearest = self._get_nearest_centerline_tangent(ego_xy)
        if nearest is None:
            print("[DEBUG] No nearest centerline found!")
            return
            
        center_xy, center_heading = nearest
        self._draw_arrow_on_canvas(canvas, center_xy[0], center_xy[1], center_heading, color=(255, 128, 0), length=15.0, width=5)

    def _draw_reference_lines_on_canvas(self):
        """Draws the reference lines on the top down canvas imitation nuplan_scenario_render."""
        if self.head_top_down_renderer is None or self.reference_lines is None:
            return
        
        try:
            import pygame
            import math
            canvas = self.head_top_down_renderer._frame_canvas
        except Exception:
            return

        # Nuplan uses magenta with alpha=0.2. In pygame we emulate alpha on white.
        color = (255, 150, 255) # Light magenta
        
        # Handle dict format (Pluto features) or list/array format (Nuplan)
        if isinstance(self.reference_lines, dict):
            positions = self.reference_lines.get("position", [])
            headings = self.reference_lines.get("orientation", [])
            valid_masks = self.reference_lines.get("valid_mask", [])
            
            # Use PyTorch or NumPy structure length
            num_lines = len(positions)
            for i in range(num_lines):
                # Stride of 20
                num_points = len(positions[i])
                for j in range(0, num_points, 15):
                    if len(valid_masks) > i and len(valid_masks[i]) > j and not valid_masks[i][j]:
                        continue
                        
                    x, y = positions[i][j][0], positions[i][j][1]
                    heading = headings[i][j]
                    
                    dx = math.cos(heading) * 1.5
                    dy = math.sin(heading) * 1.5
                    
                    start_pix = canvas.pos2pix(x, y)
                    end_pix = canvas.pos2pix(x + dx, y + dy)
                    
                    pygame.draw.line(canvas, color, start_pix, end_pix, 4)
                    
                    head_width = 0.8
                    perp_heading = heading + math.pi / 2
                    px1 = x + dx - math.cos(perp_heading) * head_width
                    py1 = y + dy - math.sin(perp_heading) * head_width
                    px2 = x + dx + math.cos(perp_heading) * head_width
                    py2 = y + dy + math.sin(perp_heading) * head_width
                    
                    p1_pix = canvas.pos2pix(px1, py1)
                    p2_pix = canvas.pos2pix(px2, py2)
                    
                    pygame.draw.polygon(canvas, color, [end_pix, p1_pix, p2_pix])

        else:
            # Original list/array format
            for ref_line in self.reference_lines:
                # Using the stride of 20 from nuplan
                for i in range(0, len(ref_line), 20):
                    p = ref_line[i]
                    x, y = p[0], p[1]
                    # If turning angle is represented differently, adjust it. 
                    # MetaDrive generally uses math.cos(heading) and math.sin(heading)
                    heading = p[2]
                    
                    # Nuplan vector length: 1.5
                    dx = math.cos(heading) * 1.5
                    dy = math.sin(heading) * 1.5
                    
                    start_pix = canvas.pos2pix(x, y)
                    end_pix = canvas.pos2pix(x + dx, y + dy)
                    
                    pygame.draw.line(canvas, color, start_pix, end_pix, 2)
                    
                    # Nuplan arrowhead width: 0.8 => 0.4 each side
                    head_width = 0.4
                    perp_heading = heading + math.pi / 2
                    px1 = x + dx - math.cos(perp_heading) * head_width
                    py1 = y + dy - math.sin(perp_heading) * head_width
                    px2 = x + dx + math.cos(perp_heading) * head_width
                    py2 = y + dy + math.sin(perp_heading) * head_width
                    
                    p1_pix = canvas.pos2pix(px1, py1)
                    p2_pix = canvas.pos2pix(px2, py2)
                    
                    pygame.draw.polygon(canvas, color, [end_pix, p1_pix, p2_pix])

    def _render_topdown(self, text: Optional[Union[dict, str]] = None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        Render the top-down view of the environment.
        :param text: text to show
        :return: top_down image
        """
        if self.head_top_down_renderer is None:
            from unitraj.renderer.top_down_renderer import TopDownRenderer
            self.head_top_down_renderer = TopDownRenderer(*args, **kwargs)
            
            # Monkey patch _draw to include our reference lines
            original_draw = self.head_top_down_renderer._draw
            def custom_draw(*a, **kw):
                original_draw(*a, **kw)
                
                # Draw our stuff on _frame_canvas (which was just blitted to screen, we need to draw and blit again)
                self._draw_reference_lines_on_canvas()
                # if hasattr(self.head_top_down_renderer, '_frame_canvas'):
                #     self._draw_diagnostics_on_canvas(self.head_top_down_renderer._frame_canvas)
                
                # RE-BLIT to _screen_canvas so our drawings are visible!
                v = self.head_top_down_renderer.current_track_agent
                canvas = self.head_top_down_renderer._frame_canvas
                screen_canvas = self.head_top_down_renderer._screen_canvas
                field = screen_canvas.get_size()
                
                if not self.head_top_down_renderer.target_agent_heading_up:
                    if self.head_top_down_renderer.position is not None or v is not None:
                        cam_pos = (self.head_top_down_renderer.position or v.position)
                        position = canvas.pos2pix(*cam_pos)
                    else:
                        position = (field[0] / 2, field[1] / 2)
                    off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
                    screen_canvas.blit(source=canvas, dest=(0, 0), area=(off[0], off[1], field[0], field[1]))
                else:
                    color_white = (255, 255, 255)
                    position = canvas.pos2pix(*v.position)
                    area = (
                        position[0] - self.head_top_down_renderer.canvas_rotate.get_size()[0] / 2, position[1] - self.head_top_down_renderer.canvas_rotate.get_size()[1] / 2,
                        self.head_top_down_renderer.canvas_rotate.get_size()[0], self.head_top_down_renderer.canvas_rotate.get_size()[1]
                    )
                    self.head_top_down_renderer.canvas_rotate.fill(color_white)
                    self.head_top_down_renderer.canvas_rotate.blit(source=canvas, dest=(0, 0), area=area)

                    import pygame
                    rotation = -np.rad2deg(v.heading_theta) + 90
                    new_canvas = pygame.transform.rotozoom(self.head_top_down_renderer.canvas_rotate, rotation, 1)

                    size = screen_canvas.get_size()
                    screen_canvas.blit(
                        new_canvas,
                        (0, 0),
                        (
                            new_canvas.get_size()[0] / 2 - size[0] / 2,
                            new_canvas.get_size()[1] / 2 - size[1] / 2,
                            size[0],
                            size[1]
                        )
                    )

            self.head_top_down_renderer._draw = custom_draw
            
        return self.head_top_down_renderer.render(text, *args, **kwargs)

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param text: text to show
        :param mode: start_top_down rendering candidate parameter is ["top_down", "topdown", "bev", "birdview"]
        :return: None or top_down image
        """

        if mode in ["top_down", "topdown", "bev", "birdview"]:
            ret = self._render_topdown(text=text, *args, **kwargs)
            return ret
        return None

    def reset(self):
        """
        Reset the renderer.
        This is a placeholder function for compatibility.
        """
        if self.head_top_down_renderer is not None:
            self.head_top_down_renderer.close()
