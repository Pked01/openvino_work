"""
 OpenVINO Profiler
 Interface for configuration classes

 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


class IConfig:
    def __init__(self, session_id: str = None, previous_job_id: int = None):
        self.session_id = session_id
        self.previous_job_id = previous_job_id

    def json(self) -> dict:
        NotImplementedError('The function json for {} is not implemented'.format(self.__class__))
