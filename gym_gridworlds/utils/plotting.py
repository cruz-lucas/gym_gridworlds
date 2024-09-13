import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from typing import Callable, List, Dict, Optional, Tuple, Any

def plot_values(
        results: Dict[Any, Dict[Any, Dict[str, np.ndarray]]],
        cmap_name: str = 'viridis', 
        norm_range: Tuple[int, int] = (-10, 1), 
        title: str = "State Values", 
        value_shape: Tuple[int, int] = (3, 3),
        aspect: str = 'equal',
        is_state_values: bool = True
) -> None:
    """Plots the state values or action-state values across different gammas and initialization values.

    Args:
        results (Dict[Any, Dict[Any, Dict[str, np.ndarray]]]): Dictionary with gamma, init_value, and results.
        cmap_name (str, optional): Name of the colormap to use. Defaults to 'viridis'.
        norm_range (Tuple[int, int], optional): Normalization range for colormap. Defaults to (-10, 1).
        title (str, optional): Title of the plot. Defaults to "State Values".
        value_shape (Tuple[int, int], optional): Shape of the value array for plotting. Defaults to (3, 3).
        aspect (str, optional): Aspect ratio of the plot. Defaults to 'equal'.
        is_state_values (bool, optional): If True, plots state values; if False, plots action-state values. Defaults to True.
    """
    gammas = list(results.keys())
    init_values = list(next(iter(results.values())).keys())

    if is_state_values:
        fig, axs = plt.subplots(len(init_values) * 2, len(gammas), figsize=(9, 19),
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.5})
        fig.suptitle(title, fontsize=24, y=.92)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=norm_range[0], vmax=norm_range[1])


    for row, init_value in enumerate(init_values):
        if is_state_values:
            axs[row * 2][0].set_ylabel(f"$V_0$: {init_value}", fontsize=18)
            axs[row * 2 + 1][0].set_ylabel(f"$V_0$: {init_value}", fontsize=18)
        else:
            fig, axs = plt.subplots(6, len(gammas), figsize=(9, 19),
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.5})
        
        for col, gamma in enumerate(gammas):
            values = results[gamma][init_value]['values']
            errors = results[gamma][init_value]['errors']
            
            if is_state_values:
                mappable = axs[row * 2][col].imshow(values.reshape(value_shape), cmap=cmap, norm=norm, aspect=aspect)
                axs[row * 2 + 1][col].plot(errors, label=f'$\gamma$ = {gamma}')
                axs[row * 2 + 1][col].set_title('Errors')
                axs[row * 2 + 1][col].legend()
                axs[row * 2][col].set_title(f'$\gamma$ = {gamma}')
                
                for (x, y), label in np.ndenumerate(values.reshape(value_shape)):
                    color = cmap(norm(label))
                    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    text_color = 'black' if brightness > 0.5 else 'white'
                    axs[row * 2][col].text(y, x, "%.2f" % label, ha='center', va='center', color=text_color)
            else:                        
                fig.suptitle(f"{title} | $V_0 = {init_value}$", fontsize=24, y=.92)
                for a in range(values.shape[1]):  # values.shape = (n_states, n_actions)
                    mappable = axs[a][col].imshow(values[:, a].reshape(value_shape), cmap=cmap, norm=norm, aspect=aspect)
                    axs[a][col].set_title(f'Action {a} - $\gamma$ = {gamma}')
                    
                    for (x, y), label in np.ndenumerate(values[:, a].reshape(value_shape)):
                        color = cmap(norm(label))
                        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                        text_color = 'black' if brightness > 0.5 else 'white'
                        axs[a][col].text(y, x, f"{label:.2f}", ha='center', va='center', color=text_color)
                    
                axs[-1][col].plot(errors, label=f'$\gamma$ = {gamma}')
                axs[-1][col].set_title('Errors')
                axs[-1][col].legend()
            
        if not is_state_values:
            plt.colorbar(mappable, ax=axs[:, :], orientation='vertical', fraction=0.05, pad=0.05)
            plt.show()
    
    if is_state_values:
        plt.colorbar(mappable, ax=axs[:, :], orientation='vertical', fraction=0.05, pad=0.05)
        plt.show()


def plot_policy(policy: np.ndarray, title: str):
    fig, axs = plt.subplots(1)
    axs.set_title(title)
    mappable = axs.imshow((policy).transpose(), cmap="binary")
    axs.set_xlabel('States')
    axs.set_ylabel('Actions')
    plt.colorbar(mappable, ax=axs, orientation='vertical', fraction=0.05, pad=0.05, shrink=0.67)
    plt.show()