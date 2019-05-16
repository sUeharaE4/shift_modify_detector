from logging import StreamHandler, Formatter, getLogger, INFO

loggers = {}


class ModifyLogger:
    """
    shift_mod_detector の共通logger.

    Attributes
    ----------

    """

    @classmethod
    def setLevelUtil(cls, logger, level):
        """
        loggerとhandlerに同じレベルを設定する.

        Parameters
        ----------
        logger : logging.Logger
            レベルを設定したいlogger.
        level : int
            設定したいレベル.

        Returns
        -------
        logger : logging.Logger
            レベルを設定したlogger.
        """
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    @classmethod
    def create_logger(cls, name=None, log_level=INFO):
        """
        loggerとを生成する.

        Parameters
        ----------
        name : str
            loggerのNamespase用文字列.
        level : int
            設定したいレベル.

        Returns
        -------
        logger : logging.Logger
            生成したlogger.
        """
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
