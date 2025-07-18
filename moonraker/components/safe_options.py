# Support for option secrets from database
#
# Copyright (C) 2025 CreatBot LYN <yinyueguodong@foxmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations
import libnacl
import libnacl.sealed

from ..common import SqlTableDefinition
from ..utils import ServerError

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper
    from .database import MoonrakerDatabase, DBProviderWrapper
    from .template import TemplateFactory


OPTIONS_TABLE = "safe_options"


def _generate_seed() -> str:
    seed = libnacl.randombytes(libnacl.randombytes_SEEDBYTES)
    return seed.hex()


def _check_options_format(options: Dict) -> Dict:
    ''' Inspection data format '''
    for key, value in options.items():
        # check type
        if not isinstance(value, (list, tuple, dict)):
            raise ServerError("value must be list, tuple or dict")

        # convert a dictionary to a tuple
        if isinstance(value, dict):
            if {"encrypt", "data"} <= value.keys():
                value = (value["encrypt"], value["data"])
                options[key] = value
            else:
                raise ServerError(
                    "value (dict type) need 2 key: ['encrypt', 'data']"
                )

        # check value format
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ServerError("value need 2 data: (bool, any)")
            if not isinstance(value[0], bool):
                raise ServerError(
                    "the frist value must be bool "
                    "which indicates whether the data is encrypted"
                )
            if value[0]:
                if not isinstance(value[1], str):
                    raise ServerError(
                        "the second value must be str while the data is encrypted"
                    )
                try:
                    bytes.fromhex(value[1])
                except ValueError:
                    raise ServerError(
                        "the encrypted data must be a hex string"
                    )

    return options


class OptionSqlDefinition(SqlTableDefinition):
    '''
    ### Secure configuration storage table with cryptographic capabilities.

    #### Table Schema:

    | Column  | Type  | Constraints | Description                     |
    | ------- | ----- | ----------- | ------------------------------- |
    | section | TEXT  | PRIMARY KEY | Configuration section name      |
    | data    | JSONB | NOT NULL    | Key-value pairs with encryption |
    | seed    | TEXT  | NOT NULL    | 64-char HEX encoded seed        |

    #### Data Format:

    - ``data`` column stores JSON with structure:
        ``{key: (encrypt: bool, data: Any)}``

    - Example:
        ```
        {
        "password": (True, "A1B2C3..."),
        "timeout": (False, 30)
        }
        ```

    #### Encryption Protocol (Curve25519):

    - Key Generation:
        - Seed: 32-byte cryptographically random
        - HEX stored as 64-character lowercase string
    - Data Encryption:
        - When encrypt=True, data is:
            - Encrypted using seed-derived key
            - Stored as HEX string
        - Plaintext values stored in native format
    '''

    name = OPTIONS_TABLE
    prototype = (
        f"""
        {OPTIONS_TABLE}(
            section TEXT PRIMARY KEY NOT NULL,
            data pyjson NOT NULL,
            seed TEXT NOT NULL
        )
        """
    )
    version = 1

    def migrate(self, last_version: int, db_provider: DBProviderWrapper) -> None:
        return


class SafeOptions:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self.values: Dict[str, Any] = {}

        # load database
        database: MoonrakerDatabase = self.server.load_component(config, 'database')
        self.option_table = database.register_table(OptionSqlDefinition())
        self._init_sync_database()

        # register to template after load 'database' component
        template: TemplateFactory = self.server.load_component(config, 'template')
        template.add_environment_global('database', self)

    async def component_init(self) -> None:
        # sync database again. (may not be necessary)
        await self._sync_database()

    def _parse_database(self, record: Dict) -> None:
        for section, options in record.items():
            assert section == options["section"]
            parsedOptions: Dict[str, Any] = {}

            seed = options["seed"]
            rawData = options["data"]
            pubkey, prikey = libnacl.crypto_box_seed_keypair(bytes.fromhex(seed))
            box = libnacl.sealed.SealedBox(pubkey, prikey)

            if isinstance(rawData, dict):
                for key, value in rawData.items():
                    if value[0]:
                        ciphertext = bytes.fromhex(value[1])
                        if len(ciphertext) >= libnacl.crypto_box_SEALBYTES:
                            try:
                                parsedOptions[key] = box.decrypt(ciphertext).decode()
                            except libnacl.CryptError:
                                parsedOptions[key] = ""
                        else:
                            parsedOptions[key] = ""
                    else:
                        parsedOptions[key] = value[1]

            parsedOptions["__seed__"] = seed
            parsedOptions["__data__"] = rawData
            parsedOptions["__pubkey__"] = pubkey.hex()

            self.values[section] = parsedOptions

    def _init_sync_database(self) -> None:
        '''
        Must be called before component_init (database operations are synchronous).
        '''
        if self.option_table._db_provider.is_alive():
            raise self.server.error(
                "Cannot parse safe_options while the eventloop is running"
            )

        cursor = self.option_table.execute(f"SELECT * FROM {OPTIONS_TABLE}").result()
        record = cursor.fetchall().result()
        storeValues = {row[0]: dict(row) for row in record}
        self._parse_database(storeValues)

    async def _sync_database(self) -> None:
        cursor = await self.option_table.execute(f"SELECT * FROM {OPTIONS_TABLE}")
        record = await cursor.fetchall()
        storeValues = {row[0]: dict(row) for row in record}
        self._parse_database(storeValues)

    async def init_section(self, section: str, options: Dict) -> None:
        if section not in self.values:
            options = _check_options_format(options)
            seed = _generate_seed()

            section_data = (section, options, seed)
            async with self.option_table as tx:
                await tx.execute(
                    f"REPLACE INTO {OPTIONS_TABLE} VALUES(?, ?, ?)",
                    section_data
                )
            await self._sync_database()

    async def update_options(self, section: str, options: Dict) -> None:
        if section not in self.values:
            raise self.server.error("init section first")
        else:
            options = _check_options_format(options)

            async with self.option_table as tx:
                await tx.execute(
                    f"UPDATE {OPTIONS_TABLE} SET data = ? WHERE section = ?",
                    (options, section)
                )

            await self._sync_database()

    async def update_option(self, section: str, key: str, value: Any) -> None:
        if section not in self.values:
            raise self.server.error("init section first")
        else:
            options = self.values[section]["__data__"]
            options[key][1] = value
            options = _check_options_format(options)

            async with self.option_table as tx:
                await tx.execute(
                    f"UPDATE {OPTIONS_TABLE} SET data = ? WHERE section = ?",
                    (options, section)
                )
            await self._sync_database()

    def has_section(self, section: str) -> bool:
        return section in self.values

    def get_seek(self, section: str) -> str:
        return self.values.get(section, {}).get("__seed__", "")

    def get_pubkey(self, section: str) -> str:
        return self.values.get(section, {}).get("__pubkey__", "")

    def __getitem__(self, key: str) -> Any:
        return self.values.get(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def load_component(config: ConfigHelper) -> SafeOptions:
    return SafeOptions(config)
