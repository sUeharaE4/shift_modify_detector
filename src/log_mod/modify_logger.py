from logging import StreamHandler, Formatter, getLogger, INFO

loggers = {}


class ModifyLogger:

    @classmethod
    def setLevelUtil(cls, logger, level):
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    @classmethod
    def create_logger(cls, name=None, log_level=INFO):
        global loggers
        if name is None:
            name = __name__

        if loggers.get(name):
            logger = loggers.get(name)
            logger = cls.setLevelUtil(logger, log_level)
            return logger

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
