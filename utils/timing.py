""" Timing Utilities
"""
from typing import Optional
import time
import datetime

import logging
logger = logging.getLogger(__name__)
log = logger


def td_to_days_hours_minutes_str(td):
    ''' Converts time delta to string '''
    # inspired by https://stackoverflow.com/questions/538666/format-timedelta-to-string
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    out = f'{td.days}D {hours:02}:{minutes:02}:{seconds:02}' if td.days else f'{hours:02}:{minutes:02}:{seconds:02}'
    return out


class Duration(object):
    ''' Base Class for progress reporting'''

    def __init__(self, totalN: int, startX: int) -> None:

        self.start_time = time.time()  # time in seconds (float)
        self.start_date = datetime.datetime.now()  # local time

        self.totalN = totalN  # total size of element to time
        self.startX = startX  # index to start from (in case of offset)

    def reset(self, totalN: int = -1, startX: int = -1) -> None:
        ''' Resets timer internals '''

        self.start_time = time.time()
        self.start_date = datetime.datetime.now()

        self.totalN = totalN if totalN != -1 else self.totalN
        self.startX = startX if startX != -1 else self.startX

    def clock(self, idx: int):
        ''' Clocks time. Takes a snapshot of timer information '''

        delta_time = time.time() - self.start_time  # delta in seconds
        delta_sample = idx - self.startX  # delta in samples

        samples_per_second = float(delta_sample) / delta_time

        duration_estimate = delta_time * float(self.totalN) / float(delta_sample)  # total duration estimate in second
        remainder_estimate = duration_estimate - delta_time  # total remainder estimated in seconds

        duration_td = datetime.timedelta(seconds=duration_estimate)  # total duration in datetime format
        remainder_td = datetime.timedelta(seconds=remainder_estimate)  # total remainder in date time format

        end_date = duration_td + self.start_date  # total ETA

        return delta_sample, delta_time, samples_per_second, duration_estimate, remainder_estimate, duration_td, remainder_td, end_date

    def clock_str(self, idx: int, zero_based: bool = False) -> str:
        ''' Returns string representation of timer state '''

        base_shift = 0 if zero_based else 1

        delta_sample, delta_time, samples_per_second, duration_estimate, remainder_estimate, duration_td, remainder_td, end_date = self.clock(idx)

        curr_idx = idx + base_shift
        perc_done = float(curr_idx) * 100.0 / self.totalN
        duration_str = td_to_days_hours_minutes_str(duration_td)
        remainder_str = td_to_days_hours_minutes_str(remainder_td)
        eta_str = str(end_date)

        out_str = f'[{curr_idx}/{self.totalN}] {perc_done:.1f}% epoch -- [{delta_sample}it/{delta_time:.2f}s] {samples_per_second:.2f}it/s -- Duration: {duration_str} - Remainder: {remainder_str} - ETA: {eta_str}'

        return out_str


class Progress(Duration):
    ''' Class to report progress of loops
        This is a no thrills, "cluster-friendly" progress reporter for computing environments
        where stdout, stderr outputs are often captured and saved to files and do not allow for
        terminal capabilities.
    '''

    def __init__(self,
                 N: int,  # total length of timer
                 percent: float = 1.0,  # 1% (percent(default
                 everyN: Optional[int] = None,
                 zero_based: bool = True,
                 logger=None,
                 name: Optional[str] = '',
                 iteration: int = 0):

        super(Progress, self).__init__(N, iteration)

        if logger is None:
            self.log = print
        else:
            self.log = logger.info

        self.percent = None
        self.everyN = None
        self.zero_based = zero_based
        self.name = name
        self.iteration = iteration  # iteration base

        if self.totalN < 1:
            raise RuntimeError(f'# Timer length N={self.totalN} cannot be < 1 (negative or zero)')

        # parse options
        if everyN is not None:
            self.percent = float(everyN) * 100.0 / float(self.totalN)
            self.everyN = everyN
        elif percent is not None:
            self.percent = percent
            self.everyN = round(float(percent) * self.totalN / 100.0)
        else:  # both None (rare case the user set both to None)
            self.percent = 1.0  # default to 1%
            self.everyN = round(float(percent) * self.totalN / 100.0)
        # minimum value
        if self.everyN == 0:
            self.everyN = 1

    def status(self) -> None:
        ''' Logs state of timer progress'''

        name_str = f'[{self.name}] ' if self.name else ''
        self.log(f'# {name_str}Progress: every {self.percent:.2f}% [{self.everyN}/{self.totalN}]')

    def progress(self, idx, header='# ') -> None:
        ''' Marks progress for time index idx '''

        if self.zero_based:
            if idx == 0:
                self.status()
            elif (idx > 0) and ((idx + 1) % self.everyN == 0) or ((idx + 1) == self.totalN):
                log_str = self.clock_str(idx, zero_based=True)
                name_str = f'[{self.name}] ' if self.name else ''
                myheader = f'# {name_str}{header}'
                self.log(f'{myheader}{log_str}')
            else:
                return
        else:  # 1-based
            if idx == 1:
                self.status()
            elif (idx > 1) and ((idx) % self.everyN == 0) or ((idx) == self.totalN):
                log_str = self.clock_str(idx, zero_based=False)
                name_str = f'[{self.name}] ' if self.name else ''
                myheader = f'# {name_str}{header}'
                self.log(f'{myheader}{log_str}')
            else:
                return
