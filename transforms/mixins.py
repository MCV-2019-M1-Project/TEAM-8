from itertools import chain


class BaseTransformMixin:
    def validate(self, data):
        return all(
            keyword in data for keyword in getattr(self, "required_keywords", [])
        )


class MultiTransformMixin:
    def _getkeywords(self):
        return getattr(self, "multi_keys", [self.multi_key])

    def apply(self, data):
        item = data[self._getkeywords()[0]]
        iters = (data[key] for key in self._getkeywords())
        if type(item) == list:
            list_of_dicts = [self.apply_single(data, *args) for args in zip(*iters)]
            keys = set(chain(list(dic.keys()) for dic in list_of_dicts))
            return {key: [dic.get(key, None) for dic in list_of_dicts] for key in keys}
        return self.apply_single(data, *iters)
