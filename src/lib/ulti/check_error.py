from src.lib.ulti import CheckRes


class CheckError(Exception):
    """

    CheckError. Used in losses.LossBase, metrics.MetricBase.
    """

    def __init__(self, check_res: CheckRes, func_signature: str):
        errs = [f"Problems occurred when calling `{func_signature}`"]

        if check_res.varargs:
            errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
        if check_res.missing:
            errs.append(f"\tmissing param: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\tduplicated param: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\tunused param: {check_res.unused}")

        Exception.__init__(self, "\n".join(errs))

        self.check_res = check_res
        self.func_signature = func_signature
