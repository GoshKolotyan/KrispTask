if __name__ == "__main__":
    from configs import ArmenianAudioDatasetConfig
    from trainer import ArmenianASRTainer
    configs = ArmenianAudioDatasetConfig()

    ASR = ArmenianASRTainer(configs=configs)

    ASR.build_pipline()