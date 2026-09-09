"""Provider-layout adapters for the GROWER M&V pipeline.

The layout classes in this module only normalise provider-specific column
names and timestamp formats.  Ground-truth snapshot cleaning, episode
construction, aggregation, and validation are shared by every layout through
``BaseMVPipeline`` and ``utils.ground_truth``.
"""

import json
import re

import pandas as pd

from base_mvpipeline import BaseMVPipeline


def _safe_outage_point(value):
    """Return an outage-point mapping without failing on malformed source rows."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value.replace("'", '"'))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


class GA1TX8(BaseMVPipeline):
    """Layout 1 (Georgia), layout 8 (Texas), and compatible Palo Alto data."""

    ground_truth_event_id_columns = ('outageRecID',)
    ground_truth_event_id_output_column = 'outageRecID'
    ground_truth_start_column = 'outageStartTime'
    ground_truth_customers_column = 'customersOutNow'
    ground_truth_utility_column = 'EMC'
    ground_truth_longitude_column = 'long'
    ground_truth_latitude_column = 'lat'
    ground_truth_zip_column = 'zip'
    ground_truth_restored_column = 'customersRestored'

    def _transform_per_county(self):
        super()._transform_per_county()
        self._per_county.rename(
            columns={'name': 'county', 'EMC': 'utilityProvider'}, inplace=True
        )
        self._per_county['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_county['timestamp']
        )

    def _transform_per_outage(self):
        super()._transform_per_outage()
        self._per_outage['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_outage['timestamp']
        )
        self._per_outage['outageStartTime'] = self._coerce_to_eastern_datetime(
            self._per_outage['outageStartTime']
        )
        self._per_outage['zip'] = self._per_outage['zip'].replace(
            ['unknown', 'Outage scale too large to extract zipcodes'], pd.NA
        )

        outage_points = self._per_outage['outagePoint'].apply(_safe_outage_point)
        self._per_outage['lat'] = outage_points.apply(lambda point: point.get('lat'))
        self._per_outage['long'] = outage_points.apply(lambda point: point.get('lng'))


class GA11TX12(BaseMVPipeline):
    """Layout 11 (Georgia) and layout 12 (Texas)."""

    ground_truth_event_id_columns = ('outage_id',)
    ground_truth_event_id_output_column = 'outage_id'
    ground_truth_start_column = 'start_time'
    ground_truth_customers_column = 'customer_affected'
    ground_truth_source_duration_column = 'duration'
    ground_truth_utility_column = 'EMC'
    ground_truth_longitude_column = 'lon'
    ground_truth_latitude_column = 'lat'
    ground_truth_zip_column = 'zipcode'

    def _transform_per_county(self):
        super()._transform_per_county()
        self._per_county.rename(
            columns={
                'Name': 'county',
                'out': 'customersOutNow',
                'count': 'customersServed',
                'EMC': 'utilityProvider',
            },
            inplace=True,
        )
        self._per_county['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_county['timestamp']
        )

    @staticmethod
    def _start_date_with_year(start_value, snapshot_timestamp):
        """Attach the snapshot year to source strings such as ``03/15 04:28 pm``."""
        if pd.isna(start_value) or pd.isna(snapshot_timestamp):
            return pd.NaT

        match = re.fullmatch(
            r'\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})\s*([ap]m)\s*',
            str(start_value),
            flags=re.IGNORECASE,
        )
        if match is None:
            return pd.NaT

        month, day, hour, minute, meridiem = match.groups()
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        if meridiem.lower() == 'am':
            hour = 0 if hour == 12 else hour
        elif hour != 12:
            hour += 12

        snapshot = pd.Timestamp(snapshot_timestamp)
        year = snapshot.year - 1 if snapshot.month == 1 and month == 12 else snapshot.year
        try:
            return pd.Timestamp(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                tz=snapshot.tz,
            )
        except ValueError:
            return pd.NaT

    def _transform_per_outage(self):
        super()._transform_per_outage()
        self._per_outage.rename(
            columns={
                'incident_id': 'outage_id',
                'start_date': 'start_time',
                'zip_code': 'zipcode',
                'consumers_affected': 'customer_affected',
            },
            inplace=True,
        )

        self._per_outage['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_outage['timestamp']
        )
        self._per_outage['start_time'] = [
            self._start_date_with_year(start_value, snapshot_timestamp)
            for start_value, snapshot_timestamp in zip(
                self._per_outage['start_time'], self._per_outage['timestamp']
            )
        ]

        duration_text = self._per_outage['duration'].astype('string').str.replace(
            r'\bhr\b', 'h', regex=True
        ).str.replace(r'\bmin\b', 'm', regex=True)
        self._per_outage['duration'] = pd.to_timedelta(
            duration_text, errors='coerce'
        ).dt.total_seconds().div(60.0)


class GA3TX16(BaseMVPipeline):
    """Layout 3 (Georgia) and layout 16 (Texas)."""

    ground_truth_event_id_columns = ('CaseNumber',)
    ground_truth_event_id_output_column = 'CaseNumber'
    ground_truth_start_column = 'OutageTime'
    ground_truth_customers_column = 'CutomersAffected'
    ground_truth_utility_column = 'EMC'
    ground_truth_longitude_column = 'long'
    ground_truth_latitude_column = 'lat'

    def _transform_per_county(self):
        super()._transform_per_county()
        self._per_county.rename(
            columns={
                'CountyName': 'county',
                'CustomersAffected': 'customersOutNow',
                'CustomersServed': 'customersServed',
                'EMC': 'utilityProvider',
            },
            inplace=True,
        )
        self._per_county['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_county['timestamp']
        )

    def _transform_per_outage(self):
        super()._transform_per_outage()
        self._per_outage['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_outage['timestamp']
        )
        self._per_outage['OutageTime'] = self._coerce_to_eastern_datetime(
            self._per_outage['OutageTime']
        )
        self._per_outage.rename(columns={'X': 'long', 'Y': 'lat'}, inplace=True)


class CAInvestor(BaseMVPipeline):
    """California investor-owned utility layout."""

    # IncidentId behaves like a row position for some California feeds and can
    # change while the same outage coordinate persists.  A rounded-location key
    # is therefore used for continuity; time gaps still split later episodes.
    ground_truth_event_id_columns = ('_ca_event_key',)
    ground_truth_event_id_output_column = 'IncidentId'
    ground_truth_start_column = 'outageStartTime'
    ground_truth_customers_column = 'ImpactedCustomers'
    ground_truth_utility_column = 'UtilityCompany'
    ground_truth_longitude_column = 'long'
    ground_truth_latitude_column = 'lat'
    ground_truth_county_column = 'County'
    ground_truth_identity_is_derived = True
    ground_truth_start_time_merge_tolerance_minutes = 31.0

    def _transform_per_county(self):
        super()._transform_per_county()
        self._per_county.rename(columns={'EMC': 'utilityProvider'}, inplace=True)
        self._per_county['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_county['timestamp']
        )

    def _transform_per_outage(self):
        super()._transform_per_outage()
        self._per_outage['timestamp'] = self._coerce_to_eastern_datetime(
            self._per_outage['timestamp']
        )
        self._per_outage['outageStartTime'] = self._coerce_to_eastern_datetime(
            self._per_outage['StartDate']
        )
        self._per_outage.rename(columns={'x': 'long', 'y': 'lat'}, inplace=True)
        longitude = pd.to_numeric(self._per_outage['long'], errors='coerce')
        latitude = pd.to_numeric(self._per_outage['lat'], errors='coerce')
        utility = self._per_outage['UtilityCompany'].astype('string').str.strip()
        county = self._per_outage['County'].astype('string').str.strip()
        has_location = longitude.notna() & latitude.notna()
        location_key = (
            utility.fillna('UNKNOWN_UTILITY')
            + '::'
            + county.fillna('UNKNOWN_COUNTY')
            + '::'
            + longitude.round(5).astype('string')
            + '::'
            + latitude.round(5).astype('string')
        )
        incident_fallback = (
            utility.fillna('UNKNOWN_UTILITY')
            + '::incident::'
            + self._per_outage['IncidentId'].astype('string').fillna('UNKNOWN')
        )
        self._per_outage['_ca_event_key'] = location_key.where(
            has_location, incident_fallback
        )
