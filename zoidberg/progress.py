# -*- coding: utf-8 -*-

# Adapted from http://stackoverflow.com/a/15860757/2043465

from sys import stdout
import time


def update_progress(progress, barLength=10, ascii=False, **kwargs):
    """Displays or updates a console progress bar

    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%

    Parameters
    ----------
    progress : float
        Number between 0 and 1
    barLength : int, optional
        Length of the progress bar
    ascii : bool, optional
        If True, use '#' as the progress indicator, otherwise use a
        Unicode character (the default)

    """

    if ascii:
        cursor = "#"
    else:
        cursor = "█"

    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{prog:-<{len}}] {perc:6.2f}% {stat}".format(
        len=barLength, prog=cursor * block, perc=progress * 100, stat=status
    )

    # Secret undocumented feature
    if "zoidberg" in kwargs:
        if kwargs["zoidberg"]:
            if barLength < 40:
                barLength = 40
            if ascii:
                face = " (;,,,;) "
                ink = "#"
            else:
                face = " (°,,,°) "
                ink = "█"

            open_claw = "(\\/)"
            closed_claw = "(|)"

            if int(progress * barLength) % 2:
                left_claw = open_claw
                right_claw = closed_claw
            else:
                left_claw = closed_claw
                right_claw = open_claw

            zb = left_claw + face + right_claw
            zb_middle = int(len(zb) / 2)
            start = int(round((barLength - zb_middle) * progress))
            text = "\rProgress: [{start}{zb}{rest}] {perc:6.2f}% {stat}".format(
                start=ink * start,
                zb=zb,
                perc=progress * 100,
                rest="-" * (barLength - start - zb_middle),
                stat=status,
            )

    stdout.write(text)
    stdout.flush()


def format_time(secs):
    ret = ""
    out = False
    for name, val in [("m", 60), ("h", 3600), ("d", 60 * 60 * 24)][::-1]:
        if secs >= val:
            out = True
        if out:
            cur = int(secs // val)
            ret += f"{cur:02d}{name}"
            secs -= cur * val
    ret += f"{secs:02.0f}s"
    return ret


class Progress(object):
    def __init__(self):
        pass

    def update(self, prog):
        needed = time.time() - self.t0
        print(
            f"{prog * 100:6.2f} % ... {format_time(needed)}>{format_time(needed / (prog) - needed) if prog else 0}",
            end="\r",
        )

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.update(1)
        print()
