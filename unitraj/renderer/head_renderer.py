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

    def set_reference_lines(self, ref_lines):
        """Set reference lines to be drawn on the next render call."""
        self.reference_lines = ref_lines

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
                for j in range(0, num_points, 5):
                    if len(valid_masks) > i and len(valid_masks[i]) > j and not valid_masks[i][j]:
                        continue
                        
                    x, y = positions[i][j][0], positions[i][j][1]
                    heading = headings[i][j]
                    
                    dx = math.cos(heading) * 1.5
                    dy = math.sin(heading) * 1.5
                    
                    start_pix = canvas.pos2pix(x, y)
                    end_pix = canvas.pos2pix(x + dx, y + dy)
                    
                    pygame.draw.line(canvas, color, start_pix, end_pix, 6)
                    
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
                self._draw_reference_lines_on_canvas()
                original_draw(*a, **kw)
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
