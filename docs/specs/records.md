# Specification: record schema

Status: draft 0.1, released with qrq 0.2.0. Point records (section 2) are
implemented and normative. Place records, sidecars and manifests (sections 3
to 5) are specified for qrq 0.3 and are not implemented in 0.2.0.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Conventions shared by all record kinds

* Coordinates are `x`, `y` in metres in one projected metric CRS per dataset.
  The CRS is stated once in the dataset manifest (section 5), never per row.
* Times are float64 seconds. Point records count from the start of the index
  day; place validity times count from the dataset epoch stated in the
  manifest (ISO 8601 with an explicit UTC offset).
* `entity` is an opaque proxy. Loaders MUST NOT store names, account
  identifiers or other direct identifiers of people in any record field.
* `source_ordinal` is strictly increasing within a source file and makes
  every ordering deterministic.
* Every dataset MUST carry licence and attribution information (section 6).

## 2. Point records (implemented)

One observed position of one entity at one time, stored as the 40-byte
structured dtype `qrq.records.POINT`:

| field | dtype | meaning |
|---|---|---|
| `x` | `<f8` | projected easting in metres |
| `y` | `<f8` | projected northing in metres |
| `t` | `<f8` | seconds since the start of the index day |
| `entity` | `<i8` | opaque entity proxy |
| `source_ordinal` | `<u8` | strictly increasing row ordinal within the source |

This layout is frozen for the 0.x series. Per-record semantic values for
points (activity, transport mode, ...) are stored as sidecars (section 4),
never as extra dtype fields.

## 3. Place records (POIs)

A place is a static entity with a location, a category, optional text and a
validity interval. Place tables are columnar (Apache Parquet, one row per
place).

| column | type | required | meaning |
|---|---|---|---|
| `place_id` | string | yes | identifier stable within its source |
| `x`, `y` | float64 | yes | projected position in metres (dataset CRS) |
| `valid_from` | float64 | yes | start of validity, seconds from the epoch; `-inf` if unknown or unbounded |
| `valid_to` | float64 | yes | end of validity; `+inf` if open |
| `operating_status` | string | yes | `open`, `temporarily_closed`, `permanently_closed` or `unknown` |
| `category_system` | string | yes | taxonomy identifier, e.g. `overture`, `fsq`, `osm` |
| `category_path` | list<string> | yes | categories from the taxonomy root to the most specific label |
| `alternate_categories` | list<list<string>> | no | further category paths in the same system |
| `name` | string | no | primary display name |
| `text` | string | no | descriptive text used by text encoders |
| `opening_hours` | string | no | raw opening-hours expression as published by the source |
| `source` | string | yes | upstream source identifier |
| `license` | string | yes | SPDX licence identifier of this row, e.g. `CDLA-Permissive-2.0`, `Apache-2.0`, `CC0-1.0`, `ODbL-1.0` |
| `attribution` | string | yes | attribution text required by the row's licence |
| `source_ordinal` | uint64 | yes | strictly increasing row ordinal |

Rules:

* A place is in a space-time window when its position is inside the spatial
  extent and its validity interval intersects the time interval. The
  predicate MUST be closed on both ends, as for `SpaceTimeBox`.
* `category_path` MUST be expressed in one taxonomy per column value; mixing
  taxonomies inside one path is not allowed. Cross-walks between taxonomies
  are encoders (see `semantic.md`), not loader logic.
* `opening_hours` is kept raw. Expanding it into validity intervals is an
  encoder step, because the expansion depends on the analysis calendar.

### 3.1 Source mappings (informative)

| target column | Overture Places | Foursquare OS Places | OpenStreetMap |
|---|---|---|---|
| `place_id` | `id` | `fsq_place_id` | element type and id, e.g. `node/123` |
| `x`, `y` | point geometry, projected | `latitude`, `longitude`, projected | node position or way centroid, projected |
| `category_path` | `categories.primary` resolved in the Overture taxonomy | `fsq_category_labels` split on the hierarchy separator | key and value of the primary tag, e.g. `["amenity", "cafe"]` |
| `name` | `names.primary` | `name` | `name` tag |
| `operating_status` | `operating_status` | `permanently_closed` when `date_closed` is set, else `open` | `unknown` unless lifecycle tags state otherwise |
| `valid_to` | `+inf` | `date_closed` when set | from lifecycle tags when present |
| `opening_hours` | not provided | not provided | `opening_hours` tag |
| `license` | `sources[].license` of the record | `Apache-2.0` | `ODbL-1.0` |

Field names follow the published schemas at the time of writing; loaders
MUST pin the release they read and record it in the manifest.

## 4. Sidecars

Semantic values are stored next to the records, keyed by encoder, so that
record files never change when an encoder is added.

* Axis values: `semantic/<encoder_id>.npy`, float64, one value per record in
  record order.
* Embeddings: `embeddings/<encoder_id>.npy`, float32 matrix `(n, d)`, rows in
  record order, plus `embeddings/<encoder_id>.json` with `model_id`,
  `model_revision`, `dim`, `normalised` and the text fields used.
* Each sidecar MUST state the SHA-256 of the record file it belongs to.

## 5. Dataset manifest

`manifest.json` at the dataset root:

```json
{
  "schema_version": "1.0",
  "dataset_id": "example-places-2026-09",
  "kind": "places",
  "crs": "EPSG:32633",
  "epoch": "2026-01-01T00:00:00+00:00",
  "record_count": 123456,
  "files": {"records.parquet": "<sha256>"},
  "upstream": [{"source": "overture", "release": "<release tag>", "accessed": "2026-10-20"}],
  "licenses": ["CDLA-Permissive-2.0", "Apache-2.0"],
  "share_alike": false,
  "notices": ["NOTICE.txt"],
  "encoders": ["taxonomy-overture-v1"]
}
```

`kind` is `points` or `places`. `share_alike` MUST be `true` when any row is
under a share-alike licence (for example `ODbL-1.0`).

## 6. Licences and attribution (normative)

* Loaders MUST keep the per-row `license` and `attribution` columns for
  places and the per-source licence for points.
* When an upstream licence requires a notice file (for example the Apache
  2.0 NOTICE of Foursquare OS Places, also required for Foursquare-sourced
  rows of Overture Places), the loader MUST copy it into the dataset and list
  it under `notices`.
* A dataset that joins share-alike rows with other rows MUST set
  `share_alike: true`; tools that publish derived tables MUST surface this
  flag. Filtering rows by `license` is the supported way to build a dataset
  without share-alike obligations.
* qrq does not redistribute third-party data. Loaders read user-provided
  copies obtained under the upstream terms.
