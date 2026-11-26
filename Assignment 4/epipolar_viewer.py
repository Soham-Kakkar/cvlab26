"""Click-to-replace epipolar overlays shared by Questions 1 and 3."""

from html import escape

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure


def line_segment(line, shape):
    """Intersect ax + by + c = 0 with the image's pixel-centre rectangle."""
    line = np.asarray(line, dtype=float)
    if not np.all(np.isfinite(line)):
        return None
    scale = np.linalg.norm(line[:2])
    if scale == 0:
        return None  # e.g. clicking exactly on the epipole
    a, b, c = line / scale
    height, width = shape[:2]
    xmax, ymax = width - 1, height - 1
    points = []

    def add(x, y):
        if -1e-7 <= x <= xmax + 1e-7 and -1e-7 <= y <= ymax + 1e-7:
            point = np.clip([x, y], [0, 0], [xmax, ymax])
            if not any(np.linalg.norm(point - old) < 1e-7 for old in points):
                points.append(point)

    if abs(b) > 1e-12:
        for x in (0, xmax):
            add(x, -(a * x + c) / b)
    if abs(a) > 1e-12:
        for y in (0, ymax):
            add(-(b * y + c) / a, y)
    if len(points) < 2:
        return None
    return np.array(points[:2])


class EpipolarViewer:
    """Two original images, optionally linked to their rectified images.

    F follows x_right.T @ F @ x_left = 0 in ORIGINAL image coordinates.
    to_rect/to_orig take (points[N, 2], side), with side 0=left, 1=right.
    Keep the viewer in a notebook variable so its callbacks stay alive.
    """

    def __init__(
        self, left, right, F, *, title="Epipolar geometry",
        labels=("Image 1", "Image 2"), rectified=None,
        to_rect=None, to_orig=None,
    ):
        self.F = np.asarray(F, dtype=float)
        if self.F.shape != (3, 3) or not np.all(np.isfinite(self.F)):
            raise ValueError("F must be a finite 3 x 3 fundamental matrix")
        self.to_rect, self.to_orig = to_rect, to_orig
        self.images = [[left, right]]
        if rectified is not None:
            if to_rect is None or to_orig is None:
                raise ValueError("Linked rectification needs both coordinate transforms")
            self.images.append(list(rectified))
        self.title = title
        self.figures = []
        self.axes = np.empty((len(self.images), 2), dtype=object)
        # Size each panel from its image aspect ratio, with only a small label gap.
        # Each stage has a narrow vertical pair; Q3 switches stages with tabs.
        width, margin, label_gap = 3.6, 0.035, 0.20
        image_width = width - 2 * margin
        for row, images in enumerate(self.images):
            heights = [image_width * im.shape[0] / im.shape[1] for im in images]
            height = sum(heights) + 2 * label_gap + 2 * margin
            fig = Figure(figsize=(width, height), dpi=100)
            self.figures.append(fig)
            top = height - margin
            for side, image_height in enumerate(heights):
                top -= label_gap + image_height
                self.axes[row, side] = fig.add_axes([
                    margin / width, top / height,
                    image_width / width, image_height / height,
                ])
        self.fig = self.figures[0]
        self._widget = None
        self.managers = []
        self.message = "Click to select; right-click to clear."
        self.markers, self.lines = {}, {}
        for row in range(len(self.images)):
            for side in range(2):
                key = row, side
                ax = self.axes[key]
                ax.imshow(self.images[row][side])
                ax.set_title(labels[side], fontsize=9, pad=3)
                ax.set_axis_off()
                # Fixed limits prevent an out-of-view transformed point from zooming out.
                ax.set_autoscale_on(False)
                self.markers[key], = ax.plot(
                    [], [], "o", ms=8, color="#ff375f", mec="white", mew=1.5,
                    zorder=4,
                )
                self.lines[key], = ax.plot(
                    [], [], color="#00ffff", lw=2, zorder=3,
                )
        self.connections = [
            fig.canvas.mpl_connect("button_press_event", self._on_click)
            for fig in self.figures
        ]
        self.connection = self.connections[0]

    def widget(self):
        """Build live canvases without changing the notebook's inline backend."""
        import matplotlib
        import ipywidgets as widgets

        # ipympl selects itself globally on its first import. Restore the caller's
        # backend before constructing our own canvases, so report plots stay inline.
        backend = matplotlib.get_backend()

        # Some Windows combinations of Matplotlib and ipympl reject ipympl's
        # module backend while ipympl.__init__ calls matplotlib.use(). The
        # viewer creates explicit ipympl canvases below, so skip only that
        # package-level backend switch and leave all other use() calls intact.
        use_backend = matplotlib.use

        def use_ipympl_canvas(backend, *args, **kwargs):
            if str(backend) == "module://ipympl.backend_nbagg":
                return
            return use_backend(backend, *args, **kwargs)

        matplotlib.use = use_ipympl_canvas
        try:
            from ipympl.backend_nbagg import Canvas, FigureManager
        finally:
            matplotlib.use = use_backend
        if matplotlib.get_backend() != backend:
            # Some Matplotlib versions cannot resolve ipympl's module backend
            # through ``matplotlib.use`` once ipympl has switched the global
            # backend.  This notebook uses the inline backend for its report
            # plots, so restore that registered backend explicitly in this case.
            if str(backend).startswith("module://ipympl."):
                backend = "module://matplotlib_inline.backend_inline"
            matplotlib.use(backend)

        if self._widget is None:
            canvases = []
            for fig in self.figures:
                canvas = Canvas(fig)
                self.managers.append(FigureManager(canvas, id(fig)))
                canvas.header_visible = False
                canvas.footer_visible = False
                canvas.toolbar_visible = False
                canvas.resizable = False
                canvas.layout.width = "100%"
                canvases.append(canvas)
            if len(canvases) == 2:
                content = widgets.Tab(children=canvases)
                content.set_title(0, "Before rectification")
                content.set_title(1, "After rectification")
                # A hidden tab may not render the first requested update.
                content.observe(lambda change: self._draw(), names="selected_index")
            else:
                content = canvases[0]
            self.status_widget = widgets.HTML(value=escape(self.message))
            self._widget = widgets.VBox(
                [widgets.HTML(value=f"<b>{escape(self.title)}</b>"), content, self.status_widget],
                layout=widgets.Layout(width="380px", max_width="100%", margin="0"),
            )
        return self._widget

    def show(self):
        from IPython.display import display

        display(self.widget())

    def _draw(self):
        for fig in self.figures:
            fig.canvas.draw_idle()

    def _set_message(self, message):
        self.message = message
        if self._widget is not None:
            self.status_widget.value = escape(message)

    def clear(self):
        for artist in (*self.markers.values(), *self.lines.values()):
            artist.set_data([], [])
        self._set_message("Click to select; right-click to clear.")
        self._draw()

    def _draw_line(self, row, side, line):
        segment = line_segment(line, self.images[row][side].shape)
        if segment is not None:
            self.lines[row, side].set_data(segment[:, 0], segment[:, 1])
        return segment is not None

    def _on_click(self, event):
        positions = np.argwhere(self.axes == event.inaxes)
        if not len(positions):
            return
        # A pan/zoom click belongs to the toolbar, not to the point selector.
        if any(ax.get_navigate_mode() for ax in self.axes.flat):
            return
        if event.button == MouseButton.RIGHT:
            self.clear()
            return
        if event.button != MouseButton.LEFT or event.xdata is None or event.ydata is None:
            return
        row, side = map(int, positions[0])
        point = np.array([[event.xdata, event.ydata]])
        height, width = self.images[row][side].shape[:2]
        if not (0 <= point[0, 0] <= width - 1 and 0 <= point[0, 1] <= height - 1):
            return
        original = point if row == 0 else np.asarray(self.to_orig(point, side))
        rect = None
        if len(self.images) == 2:
            rect = point if row == 1 else np.asarray(self.to_rect(point, side))
        if not np.all(np.isfinite(original)) or (rect is not None and not np.all(np.isfinite(rect))):
            return

        self.clear()
        self.markers[0, side].set_data(original[:, 0], original[:, 1])
        x = np.r_[original[0], 1.0]
        line = (self.F if side == 0 else self.F.T) @ x
        visible = self._draw_line(0, 1 - side, line)
        if rect is not None:
            self.markers[1, side].set_data(rect[:, 0], rect[:, 1])
            for target in range(2):
                self._draw_line(1, target, [0, 1, -rect[0, 1]])
        message = f"Selected {'left' if side == 0 else 'right'}: original ({original[0, 0]:.1f}, {original[0, 1]:.1f})"
        if rect is not None:
            message += f" | rectified row {rect[0, 1]:.1f}"
        if not visible:
            message += " | original epipolar line outside image or undefined"
        self._set_message(message)
        self._draw()

    def close(self):
        """Disconnect stale callbacks when the notebook cell is rerun."""
        for fig, connection in zip(self.figures, self.connections):
            fig.canvas.mpl_disconnect(connection)
        for manager in self.managers:
            manager.destroy()
            manager.toolbar.close()
        if self._widget is not None:
            for child in self._widget.children:
                child.close()
            self._widget.close()
