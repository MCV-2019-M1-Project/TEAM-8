from os.path import Path

from methodtools import lru_cache


class Dataset:
    """
    TODO:
        - backup:
            - load idx - [stage_name, {**new_items}]
            - update idx
            - load all

        pyTables - complicated
        pickle - to many files, or can't load by index
        numpy.savez - everything has to be an array

        - compute stage by stage
    """

    pipeline = tuple()

    def __init__(self, path, extension="jpg", backup="", start_from=None):
        # TODO: allow many extensions
        self.paths = sorted(Path(path).glob("*.{extension}"))
        self.start_from = start_from
        if backup:
            self._init_backup(backup)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self._getitem_cached(idx)

    def _init_backup(self, backup_path):
        # TODO actual backup
        pass

    @lru_cache(maxsize=512)
    def _getitem_cached(self, idx):
        data = self._load(idx)
        return self._finish_processing(data)

    def _load(self, idx):
        # TODO try to load from backup
        data = {"_dataset": self, "idx": idx, "path": self.paths[idx], "_completed": []}
        return data

    def _finish_processing(self, data, end_when_data_has=None):
        to_run = self._get_remaining_stages(data["_completed"])
        for stage_cls in to_run:
            stage_instance = stage_cls(data)
            try:
                new_items = self._apply_stage(stage_instance, data)
                self._save(data["idx"], new_items, stage_cls.__name__)
            except Exception:
                print(f"Failed in {stage_cls.__name__}")
                raise

    def save(self, idx):
        # TODO:
        pass

    def _get_remaining_stages(self, completed):
        try:
            first_difference = list(
                map(lambda c, s: c == s.__name__, zip(completed, self.pipeline))
            ).index(True)
        except ValueError:
            return ()
        return self.pipeline[first_difference:]

    def _apply_stage(self, stage, data):
        if getattr(stage, "validate", None) is not None:
            assert stage.validate(data), "Validation failed"
        if getattr(stage, "pre_apply", None) is not None:
            stage.pre_apply(data)
        new_items = stage.apply(data)
        if new_items:
            data.update(new_items)
        if getattr(stage, "post_apply", None) is not None:
            stage.post_apply(data)

    def assequence(self, of=None):
        if of is not None:
            if isinstance(of, str):
                return ({of: self.__getitem__(idx)[of]} for idx in range(len(self)))
            of = set(of)
            return (
                {
                    (key, value)
                    for key, value in self.__getitem__(idx).items()
                    if key in of
                }
                for idx in range(len(self))
            )
        return (self.__getitem__(idx) for idx in range(len(self)))


class Tester:
    def __init__(
        self, score_fn, factory, confs, verbose=False, min_score=0, compare=None
    ):
        """
        score_fn - function that takes an instance of dataset and returns a score
        factory - on each call return an instance of a dataset
        confs - configuration to test
        verbose - print scores for each configuration or just for the best one
        If scores are not just positive numbers then you have to provide:
            min_score - the "-infinity score"
            compare(lower_score, higher_score) = True
        """
        self.score_fn = score_fn
        self.factory = factory
        self.confs = confs
        self.min_score = min_score
        if compare is not None:
            self.compare = compare
        else:
            self.compare = lambda x, y: x < y
        self.verbose = verbose

    def __call__(self):
        max_score = self.min_score
        max_conf = None
        for conf in self.confs:
            instance = self.factory()
            for key, value in conf:
                setattr(instance, key, value)
            if self.verbose:
                print(f"Testing {conf}...")
            new_score = eval(instance)
            if self.verbose:
                print(f"Scored {new_score}!!!")
            if self.compare(max_score, new_score):
                max_score = new_score
                max_conf = conf
        print(f"Highest score of {max_score} achieved with {max_conf}")
