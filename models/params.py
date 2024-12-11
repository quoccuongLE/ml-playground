import copy

from pydantic import BaseModel, Field


class ConfigParams(BaseModel, extra="allow"):

    def override(self, override_dict: dict | BaseModel):
        """The implementation of `override`."""
        for k, v in override_dict.items():
            if k not in self.__dict__.keys():
                super().__setattr__(k, copy.deepcopy(v))
            else:
                if isinstance(v, dict):
                    self.__dict__[k].override(v)
                elif isinstance(v, BaseModel):
                    self.__dict__[k].override(v.__dict__)
                else:
                    self.__dict__[k] = copy.deepcopy(v)
