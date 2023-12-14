import pickle
from typing import TYPE_CHECKING
from matplotlib.artist import Artist

if TYPE_CHECKING:
    from matplotlib.axes import Axes

def copy_axes(from_ax: 'Axes', to_ax: 'Axes', axis_labels: bool = False):
    """Copy the properties and contents of one axes to another."""
    # Quick and dirty way to make a copy of an Axes
    # Matplotlib does not allow artists (plot elements) belonging to multiple axes, so we'll have to "steal"
    # them from the from_ax. To preserve from_ax, a copy is needed.
    from_ax = pickle.loads(pickle.dumps(from_ax))

    to_ax.update_from(from_ax)
    to_ax.grid()
    to_ax.xaxis.update_from(from_ax.xaxis)
    to_ax.yaxis.update_from(from_ax.yaxis)
    to_ax.set_xlim(*from_ax.get_xlim())
    to_ax.set_ylim(*from_ax.get_ylim())

    elements_adders = [
        (from_ax.lines, to_ax.add_line),
        (from_ax.collections, to_ax.add_collection),
        (from_ax.containers, to_ax.add_container)
    ]

    for elements, adder in elements_adders:
        for element in elements:
            if isinstance(element, Artist):
                # Need to detach this copy from the original Axes/Figure
                element.remove()
            if hasattr(element, '_sizes') and hasattr(element, 'set_sizes'):
                element.set_sizes(element._sizes)

            # Unset coordinate transforms
            element._transform = None
            element._transformSet = False
            element.pchanged()
            element.stale = True
            adder(element)

    for axis, source, target in [('x', from_ax.xaxis, to_ax.xaxis), ('y', from_ax.yaxis, to_ax.yaxis)]:
        target.update_from(source)
        target.set_tick_params(labelsize=source._get_tick_label_size(axis))

        if axis_labels:
            label = source.get_label()
            target.set_label_text(label.get_text(), fontsize=label.get_fontsize())
