from collections.abc import Callable
import pickle
from typing import TYPE_CHECKING, Optional
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework.matplotlib import default_figure_canvas

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


def make_list_plot(
    experiment_data: ExperimentData,
    title_fn: Optional[Callable[[int], str]] = None,
    figure_idx: int = 0,
    width: float = 6.4,
    height_per_child: float = 2.4
) -> Figure:
    component_index = experiment_data.metadata['component_child_index']
    figure = Figure(figsize=[width, height_per_child * len(component_index)])
    _ = default_figure_canvas(figure)
    axs = figure.subplots(len(component_index), 1, sharex=True)
    for child_index, to_ax in enumerate(axs):
        child_data = experiment_data.child_data(component_index[child_index])
        copy_axes(child_data.figure(figure_idx).figure.axes[0], to_ax)
        if title_fn is not None:
            to_ax.set_title(title_fn(child_index))
    figure.tight_layout()
    return figure
