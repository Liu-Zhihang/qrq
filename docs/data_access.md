# Data access

The published workload uses point records from 14 public sources in 13 cities
(Beijing contributes two sources). qrq redistributes none of them. Obtain
each source from its provider and respect its licence and terms of use;
several require registration or a data-use agreement.

| source id        | city           | source                                                     | source-days used |
|------------------|----------------|------------------------------------------------------------|------------------|
| `tdrive`         | Beijing        | T-Drive taxi trajectories                                  | 7 |
| `geolife`        | Beijing        | GeoLife GPS trajectories                                   | 8 |
| `jinan`          | Jinan          | Jinan open data portal, taxi monitoring records            | 8 |
| `dublin`         | Dublin         | Google Air View mobile monitoring                          | 8 |
| `porto`          | Porto          | Porto taxi trajectories (UCI Machine Learning Repository)  | 8 |
| `montreal`       | Montreal       | MTL Trajet travel survey traces                            | 8 |
| `barcelona`      | Barcelona      | Beepath pedestrian mobility experiments                    | 8 |
| `rio`            | Rio de Janeiro | Rio de Janeiro municipal bus GPS data                      | 8 |
| `stellenbosch`   | Stellenbosch   | Stellenbosch 1 Hz GPS traces                               | 4 |
| `nyc_mat`        | New York City  | MAT multiple-aspect trajectories, New York                 | 8 |
| `paris_mat`      | Paris          | MAT multiple-aspect trajectories, Paris                    | 8 |
| `sf_cabspotting` | San Francisco  | CRAWDAD `epfl/mobility` taxi traces                        | 8 |
| `rome_taxi`      | Rome           | CRAWDAD `roma/taxi` taxi traces                            | 8 |
| `brighton_shl`   | Brighton       | Sussex-Huawei Locomotion (SHL) dataset                     | 8 |

Total: 107 source-days.

## Monitoring-only sources

Two further public mobile-monitoring sources were profiled for future work.
They are not part of any published count.

| source                | provider record                                   | licence    |
|-----------------------|---------------------------------------------------|------------|
| Zurich OpenSense      | Zenodo record 1415369 (tram-mounted ultrafine particle monitoring) | CC-BY-4.0 |
| Portici AirHeritage   | Zenodo record 13151961 (citizen-carried PM monitoring)             | CC-BY-4.0 |

## Preparing records

For each source-day the records must be

1. projected to a metric CRS appropriate for the city (metres);
2. clipped to one index day with `t` in seconds since local midnight,
   `0 <= t < 86400`;
3. written as flat `qrq.records.POINT` arrays with an int64 entity proxy and
   a strictly increasing `source_ordinal`.

The exact cleaning, projection and study-domain rules of the published run,
and the city reference points `(cx, cy)` used by `register_windows`, are
part of the restricted inputs described in `docs/reproducing.md`.
