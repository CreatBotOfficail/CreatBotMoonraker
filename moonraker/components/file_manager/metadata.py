#!/usr/bin/env python3
# GCode metadata extraction utility
#
# Copyright (C) 2020 Eric Callahan <arksine.code@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations
import json
import argparse
import re
import os
import sys
import base64
import traceback
import tempfile
import zipfile
import shutil
import uuid
import logging
from PIL import Image

# Annotation imports
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Dict,
    List,
    Tuple,
    Type,
)
if TYPE_CHECKING:
    pass

UFP_MODEL_PATH = "/3D/model.gcode"
UFP_THUMB_PATH = "/Metadata/thumbnail.png"

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("metadata")

# Regex helpers.  These methods take patterns with placeholders
# to insert the correct regex capture group for floats, ints,
# and strings:
#  Float: (%F) = (\d*\.?\d+)
#  Integer: (%D) = (\d+)
#  String: (%S) = (.+)
def regex_find_floats(pattern: str, data: str) -> List[float]:
    pattern = pattern.replace(r"(%F)", r"([0-9]*\.?[0-9]+)")
    matches = re.findall(pattern, data)
    if matches:
        # return the maximum height value found
        try:
            return [float(h) for h in matches]
        except Exception:
            pass
    return []

def regex_find_ints(pattern: str, data: str) -> List[int]:
    pattern = pattern.replace(r"(%D)", r"([0-9]+)")
    matches = re.findall(pattern, data)
    if matches:
        # return the maximum height value found
        try:
            return [int(h) for h in matches]
        except Exception:
            pass
    return []

def regex_find_float(pattern: str, data: str) -> Optional[float]:
    pattern = pattern.replace(r"(%F)", r"([0-9]*\.?[0-9]+)")
    match = re.search(pattern, data)
    val: Optional[float] = None
    if match:
        try:
            val = float(match.group(1))
        except Exception:
            return None
    return val

def regex_find_int(pattern: str, data: str) -> Optional[int]:
    pattern = pattern.replace(r"(%D)", r"([0-9]+)")
    match = re.search(pattern, data)
    val: Optional[int] = None
    if match:
        try:
            val = int(match.group(1))
        except Exception:
            return None
    return val

def regex_find_string(pattern: str, data: str) -> Optional[str]:
    pattern = pattern.replace(r"(%S)", r"(.*)")
    match = re.search(pattern, data)
    if match:
        return match.group(1).strip('"')
    return None

def regex_find_min_float(pattern: str, data: str) -> Optional[float]:
    result = regex_find_floats(pattern, data)
    return min(result) if result else None

def regex_find_max_float(pattern: str, data: str) -> Optional[float]:
    result = regex_find_floats(pattern, data)
    return max(result) if result else None


# Slicer parsing implementations
class BaseSlicer(object):
    def __init__(self, file_path: str) -> None:
        self.path = file_path
        self.header_data: str = ""
        self.footer_data: str = ""
        self.layer_height: Optional[float] = None
        self.has_m486_objects: bool = False

    def set_data(self,
                 header_data: str,
                 footer_data: str,
                 fsize: int) -> None:
        self.header_data = header_data
        self.footer_data = footer_data
        self.size: int = fsize

    def _check_has_objects(self,
                           data: str,
                           pattern: Optional[str] = None
                           ) -> bool:
        match = re.search(
            r"\n((DEFINE_OBJECT)|(EXCLUDE_OBJECT_DEFINE)) NAME=",
            data
        )
        if match is not None:
            # Objects already processed
            fname = os.path.basename(self.path)
            logger.info(
                f"File '{fname}' currently supports cancellation, "
                "processing aborted"
            )
            if match.group(1).startswith("DEFINE_OBJECT"):
                logger.info(
                    "Legacy object processing detected.  This is not "
                    "compatible with official versions of Klipper."
                )
            return False
        # Always check M486
        patterns = [r"\nM486"]
        if pattern is not None:
            patterns.append(pattern)
        for regex in patterns:
            if re.search(regex, data) is not None:
                self.has_m486_objects = regex == r"\nM486"
                return True
        return False

    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        return None

    def has_objects(self) -> bool:
        return self._check_has_objects(self.header_data)

    def parse_gcode_start_byte(self) -> Optional[int]:
        m = re.search(r"\n[MG]\d+\s.*\n", self.header_data)
        if m is None:
            return None
        return m.start()

    def parse_gcode_end_byte(self) -> Optional[int]:
        rev_data = self.footer_data[::-1]
        m = re.search(r"\n.*\s\d+[MG]\n", rev_data)
        if m is None:
            return None
        return self.size - m.start()

    def parse_first_layer_height(self) -> Optional[float]:
        return None

    def parse_layer_height(self) -> Optional[float]:
        return None

    def parse_object_height(self) -> Optional[float]:
        return None

    def parse_filament_total(self) -> Optional[float]:
        return None

    def parse_filament_weight_total(self) -> Optional[float]:
        return None

    def parse_filament_name(self) -> Optional[str]:
        return None

    def parse_filament_type(self) -> Optional[str]:
        return None

    def parse_estimated_time(self) -> Optional[float]:
        return None

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return None

    def parse_chamber_temp(self) -> Optional[float]:
        return None

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return None

    def parse_thumbnails(self) -> Optional[List[Dict[str, Any]]]:
        for data in [self.header_data, self.footer_data]:
            thumb_matches: List[str] = re.findall(
                r"; thumbnail begin[;/\+=\w\s]+?; thumbnail end", data)
            if thumb_matches:
                break
        else:
            return None
        thumb_dir = os.path.join(os.path.dirname(self.path), ".thumbs")
        if not os.path.exists(thumb_dir):
            try:
                os.mkdir(thumb_dir)
            except Exception:
                logger.info(f"Unable to create thumb dir: {thumb_dir}")
                return None
        thumb_base = os.path.splitext(os.path.basename(self.path))[0]
        parsed_matches: List[Dict[str, Any]] = []
        has_miniature: bool = False
        for match in thumb_matches:
            lines = re.split(r"\r?\n", match.replace('; ', ''))
            info = regex_find_ints(r"(%D)", lines[0])
            data = "".join(lines[1:-1])
            if len(info) != 3:
                logger.info(
                    f"MetadataError: Error parsing thumbnail"
                    f" header: {lines[0]}")
                continue
            if len(data) != info[2]:
                logger.info(
                    f"MetadataError: Thumbnail Size Mismatch: "
                    f"detected {info[2]}, actual {len(data)}")
                continue
            thumb_name = f"{thumb_base}-{info[0]}x{info[1]}.png"
            thumb_path = os.path.join(thumb_dir, thumb_name)
            rel_thumb_path = os.path.join(".thumbs", thumb_name)
            with open(thumb_path, "wb") as f:
                f.write(base64.b64decode(data.encode()))
            parsed_matches.append({
                'width': info[0], 'height': info[1],
                'size': os.path.getsize(thumb_path),
                'relative_path': rel_thumb_path})
            if info[0] == 32 and info[1] == 32:
                has_miniature = True
        if len(parsed_matches) > 0 and not has_miniature:
            # find the largest thumb index
            largest_match = parsed_matches[0]
            for item in parsed_matches:
                if item['size'] > largest_match['size']:
                    largest_match = item
            # Create miniature thumbnail if one does not exist
            thumb_full_name = largest_match['relative_path'].split("/")[-1]
            thumb_path = os.path.join(thumb_dir, f"{thumb_full_name}")
            rel_path_small = os.path.join(".thumbs", f"{thumb_base}-32x32.png")
            thumb_path_small = os.path.join(
                thumb_dir, f"{thumb_base}-32x32.png")
            # read file
            try:
                with Image.open(thumb_path) as im:
                    # Create 32x32 thumbnail
                    im.thumbnail((32, 32))
                    im.save(thumb_path_small, format="PNG")
                    parsed_matches.insert(0, {
                        'width': im.width, 'height': im.height,
                        'size': os.path.getsize(thumb_path_small),
                        'relative_path': rel_path_small
                    })
            except Exception as e:
                logger.info(str(e))
        return parsed_matches

    def parse_layer_count(self) -> Optional[int]:
        return None

    def parse_nozzle_diameter(self) -> Optional[float]:
        return None

class UnknownSlicer(BaseSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        return {'slicer': "Unknown"}

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_min_float(r"G1\sZ(%F)\s", self.header_data)

    def parse_object_height(self) -> Optional[float]:
        return regex_find_max_float(r"G1\sZ(%F)\s", self.footer_data)

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(r"M109 S(%F)", self.header_data)

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(r"M190 S(%F)", self.header_data)

    def parse_chamber_temp(self) -> Optional[float]:
        return regex_find_float(r"M191 S(%F)", self.header_data)

    def parse_thumbnails(self) -> Optional[List[Dict[str, Any]]]:
        return None

class PrusaSlicer(BaseSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        aliases = {
            'PrusaSlicer': r"PrusaSlicer\s(.*)\son",
            'SuperSlicer': r"SuperSlicer\s(.*)\son",
            'OrcaSlicer': r"OrcaSlicer\s(.*)\son",
            'MomentSlicer': r"MomentSlicer\s(.*)\son",
            'SliCR-3D': r"SliCR-3D\s(.*)\son",
            'BambuStudio': r"BambuStudio[^ ]*\s(.*)\n",
            'A3dp-Slicer': r"A3dp-Slicer\s(.*)\son",
            'CreatWare': r"CreatWare\s(.*)\son",
        }
        for name, expr in aliases.items():
            match = re.search(expr, data)
            if match:
                return {
                    'slicer': name,
                    'slicer_version': match.group(1)
                }
        return None

    def has_objects(self) -> bool:
        return self._check_has_objects(
            self.header_data, r"\n; printing object")

    def parse_first_layer_height(self) -> Optional[float]:
        # Check percentage
        pct = regex_find_float(r"; first_layer_height = (%F)%", self.footer_data)
        if pct is not None:
            if self.layer_height is None:
                # Failed to parse the original layer height, so it is not
                # possible to calculate a percentage
                return None
            return round(pct / 100. * self.layer_height, 6)
        return regex_find_float(r"; first_layer_height = (%F)", self.footer_data)

    def parse_layer_height(self) -> Optional[float]:
        self.layer_height = regex_find_float(
            r"; layer_height = (%F)", self.footer_data
        )
        return self.layer_height

    def parse_object_height(self) -> Optional[float]:
        matches = re.findall(
            r";BEFORE_LAYER_CHANGE\n(?:.*\n)?;(\d+\.?\d*)", self.footer_data)
        if matches:
            try:
                matches = [float(m) for m in matches]
            except Exception:
                pass
            else:
                return max(matches)
        return regex_find_max_float(r"G1\sZ(%F)\sF", self.footer_data)

    def parse_filament_total(self) -> Optional[float]:
        line = regex_find_string(r'filament\sused\s\[mm\]\s=\s(%S)\n', self.footer_data)
        if line:
            filament = regex_find_floats(
                r"(%F)", line
            )
            if filament:
                return sum(filament)
        return None

    def parse_filament_weight_total(self) -> Optional[float]:
        return regex_find_float(
            r"total\sfilament\sused\s\[g\]\s=\s(%F)",
            self.footer_data
        )

    def parse_filament_type(self) -> Optional[str]:
        return regex_find_string(r";\sfilament_type\s=\s(%S)", self.footer_data)

    def parse_filament_name(self) -> Optional[str]:
        return regex_find_string(
            r";\sfilament_settings_id\s=\s(%S)", self.footer_data
        )

    def parse_estimated_time(self) -> Optional[float]:
        time_match = re.search(
            r';\sestimated\sprinting\stime.*', self.footer_data)
        if not time_match:
            return None
        total_time = 0
        time_group = time_match.group()
        time_patterns = [(r"(\d+)d", 24*60*60), (r"(\d+)h", 60*60),
                         (r"(\d+)m", 60), (r"(\d+)s", 1)]
        try:
            for pattern, multiplier in time_patterns:
                t = re.search(pattern, time_group)
                if t:
                    total_time += int(t.group(1)) * multiplier
        except Exception:
            return None
        return round(total_time, 2)

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(
            r"; first_layer_temperature = (%F)", self.footer_data
        )

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(
            r"; first_layer_bed_temperature = (%F)", self.footer_data
        )

    def parse_chamber_temp(self) -> Optional[float]:
        return regex_find_float(
            r"; chamber_temperature = (%F)", self.footer_data
        )

    def parse_nozzle_diameter(self) -> Optional[float]:
        return regex_find_float(
            r";\snozzle_diameter\s=\s(%F)", self.footer_data
        )

    def parse_layer_count(self) -> Optional[int]:
        return regex_find_int(r"; total layers count = (%D)", self.footer_data)

class Slic3rPE(PrusaSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        match = re.search(r"Slic3r\sPrusa\sEdition\s(.*)\son", data)
        if match:
            return {
                'slicer': "Slic3r PE",
                'slicer_version': match.group(1)
            }
        return None

    def parse_filament_total(self) -> Optional[float]:
        return regex_find_float(r"filament\sused\s=\s(%F)mm", self.footer_data)

    def parse_thumbnails(self) -> Optional[List[Dict[str, Any]]]:
        return None

class Slic3r(Slic3rPE):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        match = re.search(r"Slic3r\s(\d.*)\son", data)
        if match:
            return {
                'slicer': "Slic3r",
                'slicer_version': match.group(1)
            }
        return None

    def parse_filament_total(self) -> Optional[float]:
        filament = regex_find_float(
            r";\sfilament\_length\_m\s=\s(%F)", self.footer_data
        )
        if filament is not None:
            filament *= 1000
        return filament

    def parse_filament_weight_total(self) -> Optional[float]:
        return regex_find_float(r";\sfilament\smass\_g\s=\s(%F)", self.footer_data)

    def parse_estimated_time(self) -> Optional[float]:
        return None

class Cura(BaseSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        match = re.search(r"Cura_SteamEngine\s(.*)", data)
        if match:
            return {
                'slicer': "Cura",
                'slicer_version': match.group(1)
            }
        return None

    def has_objects(self) -> bool:
        return self._check_has_objects(self.header_data, r"\n;MESH:")

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_float(r";MINZ:(%F)", self.header_data)

    def parse_layer_height(self) -> Optional[float]:
        self.layer_height = regex_find_float(
            r";Layer\sheight:\s(%F)", self.header_data
        )
        return self.layer_height

    def parse_object_height(self) -> Optional[float]:
        return regex_find_float(r";MAXZ:(%F)", self.header_data)

    def parse_filament_total(self) -> Optional[float]:
        filament = regex_find_float(r";Filament\sused:\s(%F)m", self.header_data)
        if filament is not None:
            filament *= 1000
        return filament

    def parse_filament_weight_total(self) -> Optional[float]:
        return regex_find_float(r";Filament\sweight\s=\s.(%F).", self.header_data)

    def parse_filament_type(self) -> Optional[str]:
        return regex_find_string(r";Filament\stype\s=\s(%S)", self.header_data)

    def parse_filament_name(self) -> Optional[str]:
        return regex_find_string(r";Filament\sname\s=\s(%S)", self.header_data)

    def parse_estimated_time(self) -> Optional[float]:
        return regex_find_max_float(r";TIME:(%F)", self.header_data)

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(r"M109 S(%F)", self.header_data)

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(r"M190 S(%F)", self.header_data)

    def parse_chamber_temp(self) -> Optional[float]:
        return regex_find_float(r"M191 S(%F)", self.header_data)

    def parse_layer_count(self) -> Optional[int]:
        return regex_find_int(r";LAYER_COUNT\:(%D)", self.header_data)

    def parse_nozzle_diameter(self) -> Optional[float]:
        return regex_find_float(r";Nozzle\sdiameter\s=\s(%F)", self.header_data)

    def parse_thumbnails(self) -> Optional[List[Dict[str, Any]]]:
        # Attempt to parse thumbnails from file metadata
        thumbs = super().parse_thumbnails()
        if thumbs is not None:
            return thumbs
        # Check for thumbnails extracted from the ufp
        thumb_dir = os.path.join(os.path.dirname(self.path), ".thumbs")
        thumb_base = os.path.splitext(os.path.basename(self.path))[0]
        thumb_path = os.path.join(thumb_dir, f"{thumb_base}.png")
        rel_path_full = os.path.join(".thumbs", f"{thumb_base}.png")
        rel_path_small = os.path.join(".thumbs", f"{thumb_base}-32x32.png")
        thumb_path_small = os.path.join(thumb_dir, f"{thumb_base}-32x32.png")
        if not os.path.isfile(thumb_path):
            return None
        # read file
        thumbs = []
        try:
            with Image.open(thumb_path) as im:
                thumbs.append({
                    'width': im.width, 'height': im.height,
                    'size': os.path.getsize(thumb_path),
                    'relative_path': rel_path_full
                })
                # Create 32x32 thumbnail
                im.thumbnail((32, 32), Image.Resampling.LANCZOS)
                im.save(thumb_path_small, format="PNG")
                thumbs.insert(0, {
                    'width': im.width, 'height': im.height,
                    'size': os.path.getsize(thumb_path_small),
                    'relative_path': rel_path_small
                })
        except Exception as e:
            logger.info(str(e))
            return None
        return thumbs

class Simplify3D(BaseSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        match = re.search(r"Simplify3D\(R\)\sVersion\s(.*)", data)
        if match:
            self._version = match.group(1)
            self._is_v5 = self._version.startswith("5")
            return {
                'slicer': "Simplify3D",
                'slicer_version': match.group(1)
            }
        return None

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_min_float(r"G1\sZ(%F)\s", self.header_data)

    def parse_layer_height(self) -> Optional[float]:
        self.layer_height = regex_find_float(
            r";\s+layerHeight,(%F)", self.header_data
        )
        return self.layer_height

    def parse_object_height(self) -> Optional[float]:
        return regex_find_max_float(r"G1\sZ(%F)\s", self.footer_data)

    def parse_filament_total(self) -> Optional[float]:
        return regex_find_float(
            r";\s+(?:Filament\slength|Material\sLength):\s(%F)\smm",
            self.footer_data
        )

    def parse_filament_weight_total(self) -> Optional[float]:
        return regex_find_float(
            r";\s+(?:Plastic\sweight|Material\sWeight):\s(%F)\sg",
            self.footer_data
        )

    def parse_filament_name(self) -> Optional[str]:
        return regex_find_string(
            r";\s+printMaterial,(%S)", self.header_data)

    def parse_filament_type(self) -> Optional[str]:
        return regex_find_string(
            r";\s+makerBotModelMaterial,(%S)", self.footer_data)

    def parse_estimated_time(self) -> Optional[float]:
        time_match = re.search(r';\s+Build (t|T)ime:.*', self.footer_data)
        if not time_match:
            return None
        total_time = 0
        time_group = time_match.group()
        time_patterns = [(r"(\d+)\shours?", 60*60), (r"(\d+)\smin", 60),
                         (r"(\d+)\ssec", 1)]
        try:
            for pattern, multiplier in time_patterns:
                t = re.search(pattern, time_group)
                if t:
                    total_time += int(t.group(1)) * multiplier
        except Exception:
            return None
        return round(total_time, 2)

    def _get_temp_items(self, pattern: str) -> List[str]:
        match = re.search(pattern, self.header_data)
        if match is None:
            return []
        return match.group().split(",")[1:]

    def _get_first_layer_temp(self, heater: str) -> Optional[float]:
        heaters = self._get_temp_items(r"temperatureName.*")
        temps = self._get_temp_items(r"temperatureSetpointTemperatures.*")
        for h, temp in zip(heaters, temps):
            if h == heater:
                try:
                    return float(temp)
                except Exception:
                    return None
        return None

    def _get_first_layer_temp_v5(self, heater_type: str) -> Optional[float]:
        pattern = (
            r";\s+temperatureController,.+?"
            r";\s+temperatureType,"f"{heater_type}"r".+?"
            r";\s+temperatureSetpoints,\d+\|(\d+)"
        )
        match = re.search(pattern, self.header_data, re.MULTILINE | re.DOTALL)
        if match is not None:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        if self._is_v5:
            return self._get_first_layer_temp_v5("extruder")
        else:
            return self._get_first_layer_temp("Extruder 1")

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        if self._is_v5:
            return self._get_first_layer_temp_v5("platform")
        else:
            return self._get_first_layer_temp("Heated Bed")

    def parse_nozzle_diameter(self) -> Optional[float]:
        return regex_find_float(
            r";\s+(?:extruderDiameter|nozzleDiameter),(%F)",
            self.header_data
        )

class KISSlicer(BaseSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, Any]]:
        match = re.search(r";\sKISSlicer", data)
        if match:
            ident = {'slicer': "KISSlicer"}
            vmatch = re.search(r";\sversion\s(.*)", data)
            if vmatch:
                version = vmatch.group(1).replace(" ", "-")
                ident['slicer_version'] = version
            return ident
        return None

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_float(
            r";\s+first_layer_thickness_mm\s=\s(%F)", self.header_data)

    def parse_layer_height(self) -> Optional[float]:
        self.layer_height = regex_find_float(
            r";\s+max_layer_thickness_mm\s=\s(%F)", self.header_data)
        return self.layer_height

    def parse_object_height(self) -> Optional[float]:
        return regex_find_max_float(
            r";\sEND_LAYER_OBJECT\sz=(%F)", self.footer_data)

    def parse_filament_total(self) -> Optional[float]:
        filament = regex_find_floats(
            r";\s+Ext #\d+\s+=\s+(%F)\s*mm", self.footer_data)
        if filament:
            return sum(filament)
        return None

    def parse_estimated_time(self) -> Optional[float]:
        time = regex_find_float(
            r";\sCalculated.*Build\sTime:\s(%F)\sminutes",
            self.footer_data)
        if time is not None:
            time *= 60
            return round(time, 2)
        return None

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(r"; first_layer_C = (%F)", self.header_data)

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(r"; bed_C = (%F)", self.header_data)

    def parse_chamber_temp(self) -> Optional[float]:
        return regex_find_float(r"; chamber_C = (%F)", self.header_data)


class IdeaMaker(BaseSlicer):
    def check_identity(self, data: str) -> Optional[Dict[str, str]]:
        match = re.search(r"\sideaMaker\s(.*),", data)
        if match:
            return {
                'slicer': "IdeaMaker",
                'slicer_version': match.group(1)
            }
        return None

    def has_objects(self) -> bool:
        return self._check_has_objects(self.header_data, r"\n;PRINTING:")

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_float(
            r";LAYER:0\s*.*\s*;HEIGHT:(%F)", self.header_data
        )

    def parse_layer_height(self) -> Optional[float]:
        return regex_find_float(
            r";LAYER:1\s*.*\s*;HEIGHT:(%F)", self.header_data
        )

    def parse_object_height(self) -> Optional[float]:
        return regex_find_float(r";Bounding Box:(?:\s+(%F))+", self.header_data)

    def parse_filament_total(self) -> Optional[float]:
        filament = regex_find_floats(
            r";Material.\d\sUsed:\s+(%F)", self.footer_data
        )
        if filament:
            return sum(filament)
        return None

    def parse_filament_type(self) -> Optional[str]:
        return (
            regex_find_string(r";Filament\sType\s.\d:\s(%S)", self.header_data) or
            regex_find_string(r";Filament\stype\s=\s(%S)", self.header_data)
        )

    def parse_filament_name(self) -> Optional[str]:
        return (
            regex_find_string(r";Filament\sName\s.\d:\s(%S)", self.header_data) or
            regex_find_string(r";Filament\sname\s=\s(%S)", self.header_data)
        )

    def parse_filament_weight_total(self) -> Optional[float]:
        pi = 3.141592653589793
        length = regex_find_floats(
            r";Material.\d\sUsed:\s+(%F)", self.footer_data)
        diameter = regex_find_floats(
            r";Filament\sDiameter\s.\d:\s+(%F)", self.header_data)
        density = regex_find_floats(
            r";Filament\sDensity\s.\d:\s+(%F)", self.header_data)
        if len(length) == len(density) == len(diameter):
            # calc individual weight for each filament with m=pi/4*d²*l*rho
            weights = [(pi/4 * diameter[i]**2 * length[i] * density[i]/10**6)
                       for i in range(len(length))]
            return sum(weights)
        return None

    def parse_estimated_time(self) -> Optional[float]:
        return regex_find_float(r";Print\sTime:\s(%F)", self.footer_data)

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(r"M109 T0 S(%F)", self.header_data)

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(r"M190 S(%F)", self.header_data)

    def parse_chamber_temp(self) -> Optional[float]:
        return regex_find_float(r"M191 S(%F)", self.header_data)

    def parse_nozzle_diameter(self) -> Optional[float]:
        return regex_find_float(
            r";Dimension:(?:\s\d+\.\d+){3}\s(%F)", self.header_data)

class IceSL(BaseSlicer):
    def check_identity(self, data) -> Optional[Dict[str, Any]]:
        match = re.search(r"<IceSL\s(.*)>", data)
        if match:
            version = match.group(1) if match.group(1)[0].isdigit() else "-"
            return {
                'slicer': "IceSL",
                'slicer_version': version
            }
        return None

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_float(
            r";\sz_layer_height_first_layer_mm\s:\s+(%F)",
            self.header_data)

    def parse_layer_height(self) -> Optional[float]:
        self.layer_height = regex_find_float(
            r";\sz_layer_height_mm\s:\s+(%F)",
            self.header_data)
        return self.layer_height

    def parse_object_height(self) -> Optional[float]:
        return regex_find_float(
            r";\sprint_height_mm\s:\s+(%F)", self.header_data)

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(
            r";\sextruder_temp_degree_c_0\s:\s+(%F)", self.header_data)

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(
            r";\sbed_temp_degree_c\s:\s+(%F)", self.header_data)

    def parse_chamber_temp(self) -> Optional[float]:
        return regex_find_float(
            r";\schamber_temp_degree_c\s:\s+(%F)", self.header_data)

    def parse_filament_total(self) -> Optional[float]:
        return regex_find_float(
            r";\sfilament_used_mm\s:\s+(%F)", self.header_data)

    def parse_filament_weight_total(self) -> Optional[float]:
        return regex_find_float(
            r";\sfilament_used_g\s:\s+(%F)", self.header_data)

    def parse_filament_name(self) -> Optional[str]:
        return regex_find_string(
            r";\sfilament_name\s:\s+(%S)", self.header_data)

    def parse_filament_type(self) -> Optional[str]:
        return regex_find_string(
            r";\sfilament_type\s:\s+(%S)", self.header_data)

    def parse_estimated_time(self) -> Optional[float]:
        return regex_find_float(
            r";\sestimated_print_time_s\s:\s+(%F)", self.header_data)

    def parse_layer_count(self) -> Optional[int]:
        return regex_find_int(
            r";\slayer_count\s:\s+(%D)", self.header_data)

    def parse_nozzle_diameter(self) -> Optional[float]:
        return regex_find_float(
            r";\snozzle_diameter_mm_0\s:\s+(%F)", self.header_data)

class KiriMoto(BaseSlicer):
    def check_identity(self, data) -> Optional[Dict[str, Any]]:
        variants: Dict[str, str] = {
            "Kiri:Moto": r"; Generated by Kiri:Moto (\d.+)",
            "SimplyPrint": r"; Generated by Kiri:Moto \(SimplyPrint\) (.+)"
        }
        for name, pattern in variants.items():
            match = re.search(pattern, data)
            if match:
                return {
                    "slicer": name,
                    "slicer_version": match.group(1)
                }
        return None

    def parse_first_layer_height(self) -> Optional[float]:
        return regex_find_float(
            r"; firstSliceHeight = (%F)", self.header_data
        )

    def parse_layer_height(self) -> Optional[float]:
        self.layer_height = regex_find_float(
            r"; sliceHeight = (%F)", self.header_data
        )
        return self.layer_height

    def parse_object_height(self) -> Optional[float]:
        return regex_find_max_float(
            r"G1 Z(%F) (?:; z-hop end|F\d+\n)", self.footer_data
        )

    def parse_layer_count(self) -> Optional[int]:
        matches = re.findall(
            r";; --- layer (\d+) \(.+", self.footer_data
        )
        if not matches:
            return None
        try:
            return int(matches[-1]) + 1
        except Exception:
            return None

    def parse_estimated_time(self) -> Optional[float]:
        return regex_find_int(r"; --- print time: (%D)s", self.footer_data)

    def parse_filament_total(self) -> Optional[float]:
        return regex_find_float(
            r"; --- filament used: (%F) mm", self.footer_data
        )

    def parse_first_layer_extr_temp(self) -> Optional[float]:
        return regex_find_float(
            r"; firstLayerNozzleTemp = (%F)", self.header_data
        )

    def parse_first_layer_bed_temp(self) -> Optional[float]:
        return regex_find_float(
            r"; firstLayerBedTemp = (%F)", self.header_data
        )


READ_SIZE = 1024 * 1024  # 1 MiB
SUPPORTED_SLICERS: List[Type[BaseSlicer]] = [
    PrusaSlicer, Slic3rPE, Slic3r, Cura, Simplify3D,
    KISSlicer, IdeaMaker, IceSL, KiriMoto
]
SUPPORTED_DATA = [
    'gcode_start_byte',
    'gcode_end_byte',
    'layer_count',
    'object_height',
    'estimated_time',
    'nozzle_diameter',
    'layer_height',
    'first_layer_height',
    'first_layer_extr_temp',
    'first_layer_bed_temp',
    'chamber_temp',
    'filament_name',
    'filament_type',
    'filament_total',
    'filament_weight_total',
    'thumbnails']

def process_objects(file_path: str, slicer: BaseSlicer, name: str) -> bool:
    try:
        from preprocess_cancellation import (
            preprocess_slicer,
            preprocess_cura,
            preprocess_ideamaker,
            preprocess_m486
        )
    except ImportError:
        logger.info("Module 'preprocess-cancellation' failed to load")
        return False
    fname = os.path.basename(file_path)
    logger.info(
        f"Performing Object Processing on file: {fname}, "
        f"sliced by {name}"
    )
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_file = os.path.join(tmp_dir_name, fname)
        with open(file_path, 'r') as in_file:
            with open(tmp_file, 'w') as out_file:
                try:
                    if slicer.has_m486_objects:
                        processor = preprocess_m486
                    elif isinstance(slicer, PrusaSlicer):
                        processor = preprocess_slicer
                    elif isinstance(slicer, Cura):
                        processor = preprocess_cura
                    elif isinstance(slicer, IdeaMaker):
                        processor = preprocess_ideamaker
                    else:
                        logger.info(
                            f"Object Processing Failed, slicer {name}"
                            "not supported"
                        )
                        return False
                    for line in processor(in_file):
                        out_file.write(line)
                except Exception as e:
                    logger.info(f"Object processing failed: {e}")
                    return False
        if os.path.islink(file_path):
            file_path = os.path.realpath(file_path)
        shutil.move(tmp_file, file_path)
    return True

def get_slicer(file_path: str) -> Tuple[BaseSlicer, Dict[str, str]]:
    header_data = footer_data = ""
    slicer: Optional[BaseSlicer] = None
    size = os.path.getsize(file_path)
    with open(file_path, 'r') as f:
        # read the default size, which should be enough to
        # identify the slicer
        header_data = f.read(READ_SIZE)
        for impl in SUPPORTED_SLICERS:
            slicer = impl(file_path)
            ident = slicer.check_identity(header_data)
            if ident is not None:
                break
        else:
            slicer = UnknownSlicer(file_path)
            ident = slicer.check_identity(header_data)
        if size > READ_SIZE * 2:
            f.seek(size - READ_SIZE)
            footer_data = f.read()
        elif size > READ_SIZE:
            remaining = size - READ_SIZE
            footer_data = header_data[remaining - READ_SIZE:] + f.read()
        else:
            footer_data = header_data
        slicer.set_data(header_data, footer_data, size)
    if ident is None:
        ident = {"slicer": "unknown"}
    return slicer, ident

def extract_metadata(
    file_path: str, check_objects: bool
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    slicer, ident = get_slicer(file_path)
    if check_objects and slicer.has_objects():
        name = ident.get("slicer", "unknown")
        if process_objects(file_path, slicer, name):
            slicer, ident = get_slicer(file_path)
    metadata['size'] = os.path.getsize(file_path)
    metadata['modified'] = os.path.getmtime(file_path)
    metadata['uuid'] = str(uuid.uuid4())
    metadata.update(ident)
    for key in SUPPORTED_DATA:
        func = getattr(slicer, "parse_" + key)
        result = func()
        if result is not None:
            metadata[key] = result
    return metadata

def extract_ufp(ufp_path: str, dest_path: str) -> None:
    if not os.path.isfile(ufp_path):
        logger.info(f"UFP file Not Found: {ufp_path}")
        sys.exit(-1)
    thumb_name = os.path.splitext(
        os.path.basename(dest_path))[0] + ".png"
    dest_thumb_dir = os.path.join(os.path.dirname(dest_path), ".thumbs")
    dest_thumb_path = os.path.join(dest_thumb_dir, thumb_name)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_thumb_path = ""
            with zipfile.ZipFile(ufp_path) as zf:
                tmp_model_path = zf.extract(
                    UFP_MODEL_PATH, path=tmp_dir_name)
                if UFP_THUMB_PATH in zf.namelist():
                    tmp_thumb_path = zf.extract(
                        UFP_THUMB_PATH, path=tmp_dir_name)
            if os.path.islink(dest_path):
                dest_path = os.path.realpath(dest_path)
            shutil.move(tmp_model_path, dest_path)
            if tmp_thumb_path:
                if not os.path.exists(dest_thumb_dir):
                    os.mkdir(dest_thumb_dir)
                shutil.move(tmp_thumb_path, dest_thumb_path)
    except Exception:
        logger.info(traceback.format_exc())
        sys.exit(-1)
    try:
        os.remove(ufp_path)
    except Exception:
        logger.info(f"Error removing ufp file: {ufp_path}")

def main(path: str,
         filename: str,
         ufp: Optional[str],
         check_objects: bool
         ) -> None:
    file_path = os.path.join(path, filename)
    if ufp is not None:
        extract_ufp(ufp, file_path)
    metadata: Dict[str, Any] = {}
    if not os.path.isfile(file_path):
        logger.info(f"File Not Found: {file_path}")
        sys.exit(-1)
    try:
        metadata = extract_metadata(file_path, check_objects)
    except Exception:
        logger.info(traceback.format_exc())
        sys.exit(-1)
    fd = sys.stdout.fileno()
    data = json.dumps(
        {'file': filename, 'metadata': metadata}).encode()
    while data:
        try:
            ret = os.write(fd, data)
        except OSError:
            continue
        data = data[ret:]


if __name__ == "__main__":
    # Parse start arguments
    parser = argparse.ArgumentParser(
        description="GCode Metadata Extraction Utility")
    parser.add_argument(
        "-f", "--filename", metavar='<filename>',
        help="name gcode file to parse")
    parser.add_argument(
        "-p", "--path", default=os.path.abspath(os.path.dirname(__file__)),
        metavar='<path>',
        help="optional absolute path for file"
    )
    parser.add_argument(
        "-u", "--ufp", metavar="<ufp file>", default=None,
        help="optional path of ufp file to extract"
    )
    parser.add_argument(
        "-o", "--check-objects", dest='check_objects', action='store_true',
        help="process gcode file for exclude opbject functionality")
    args = parser.parse_args()
    check_objects = args.check_objects
    enabled_msg = "enabled" if check_objects else "disabled"
    logger.info(f"Object Processing is {enabled_msg}")
    main(args.path, args.filename, args.ufp, check_objects)
