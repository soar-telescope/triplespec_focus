import json

from .utils import get_args, setup_logging
from .triplespec_focus import TripleSpecFocus


def run_triplespec_focus(args=None):

    args = get_args(arguments=args)

    focus = TripleSpecFocus(debug=args.debug,
                            focus_key=args.focus_key,
                            filename_key=args.filename_key,
                            n_brightest=args.brightest,
                            saturation=args.saturation,
                            plot_results=args.plot_results,
                            debug_plots=args.debug_plots)

    results = focus(data_path=args.data_path,
                    source_fwhm=args.source_fwhm,
                    det_threshold=args.detection_threshold,
                    mask_threshold=args.mask_threshold,
                    n_brightest=args.brightest,
                    saturation_level=args.saturation,
                    show_mask=args.show_mask,
                    plot_results=args.plot_results,
                    debug_plots=args.debug_plots,
                    print_all_data=False)
    print(json.dumps(results, indent=4))