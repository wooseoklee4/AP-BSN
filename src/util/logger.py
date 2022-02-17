import threading
import datetime, os

from .progress_msg import ProgressMsg
# from .chart import LossChart


class Logger(ProgressMsg):
    def __init__(self, max_iter:tuple=None, log_dir:str=None, log_file_option:str='w', log_lvl:str='note', log_file_lvl:str='info', log_include_time:bool=True):
        '''
        Args:
            session_name (str)
            max_iter (tuple) : max iteration for progress
            log_dir (str) : if None, no file out for logging
            log_file_option (str) : 'w' or 'a'
            log_lvl (str) : 'debug' < 'note' < 'info' < 'highlight' < 'val'
            log_include_time (bool)
        '''
        self.lvl_list = ['debug', 'note', 'info', 'highlight', 'val']
        self.lvl_color = [bcolors.FAIL, None, None, bcolors.WARNING, bcolors.OKGREEN]

        assert log_file_option in ['w', 'a']
        assert log_lvl in self.lvl_list
        assert log_file_lvl in self.lvl_list

        # init progress message class
        ProgressMsg.__init__(self, max_iter)

        # log setting
        self.log_dir = log_dir
        self.log_lvl      = self.lvl_list.index(log_lvl)
        self.log_file_lvl = self.lvl_list.index(log_file_lvl)
        self.log_include_time = log_include_time
        
        # init logging
        if self.log_dir is not None:
            logfile_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
            self.log_file = open(os.path.join(log_dir, 'log_%s.log'%logfile_time), log_file_option)
            self.val_file = open(os.path.join(log_dir, 'validation_%s.log'%logfile_time), log_file_option)

    def _print(self, txt, lvl_n, end):
        txt = str(txt)
        if self.log_lvl <= lvl_n:
            if self.lvl_color[lvl_n] is not None:
                print('\033[K'+ self.lvl_color[lvl_n] + txt + bcolors.ENDC, end=end)
            else:
                print('\033[K'+txt, end=end)
        if self.log_file_lvl <= lvl_n:
            self.write_file(txt)

    def debug(self, txt, end=None):
        self._print(txt, self.lvl_list.index('debug'), end)
    
    def note(self, txt, end=None):
        self._print(txt, self.lvl_list.index('note'), end)

    def info(self, txt, end=None):
        self._print(txt, self.lvl_list.index('info'), end)

    def highlight(self, txt, end=None):
        self._print(txt, self.lvl_list.index('highlight'), end)

    def val(self, txt, end=None):
        self._print(txt, self.lvl_list.index('val'), end)
        if self.log_dir is not None:
            self.val_file.write(txt+'\n')
            self.val_file.flush()

    def write_file(self, txt):
        if self.log_dir is not None:
            if self.log_include_time:
                time = datetime.datetime.now().strftime('%H:%M:%S')
                txt = "[%s] "%time + txt
            self.log_file.write(txt+'\n')
            self.log_file.flush()

    def clear_screen(self):
        if os.name == 'nt': 
            os.system('cls') 
        else: 
            os.system('clear') 

# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
