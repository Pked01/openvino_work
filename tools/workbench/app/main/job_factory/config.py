"""
 OpenVINO Profiler
 Class for DB adapter for celery

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
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import NullPool

from app import get_config
from config.constants import SERVER_MODE


class CeleryDBAdapter:
    app_config = get_config()[SERVER_MODE]
    db_engine = create_engine(app_config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool)
    session = scoped_session(sessionmaker(autocommit=False, bind=db_engine))
