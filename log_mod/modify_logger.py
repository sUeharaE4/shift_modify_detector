from logging import StreamHandler, Formatter, getLogger, INFO

loggers = {}


class ModifyLogger:

    @classmethod
    def create_logger(cls, name=None, log_level=INFO):
        global loggers
        if name is None:
            name = __name__

        if loggers.get(name):
            return loggers.get(name)

        handler = StreamHandler()
        formatter = \
            Formatter('TIME:%(asctime)s%(tab)sLEVEL:%(levelname)s%(tab)s'
                      'LINES:%(lineno)d%(tab)sMESSAGE:%(message)s')
        handler.setFormatter(formatter)
        logger = getLogger(name)
        logger.setLevel(log_level)
        handler.setLevel(log_level)
        logger.addHandler(handler)
        loggers[name] = logger
        return logger
