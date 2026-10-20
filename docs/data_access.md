# Data access

The published workload uses point records from 14 public sources in 13 cities
(Beijing contributes two sources). qrq redistributes none of them. Obtain
each source from its provider and respect its licence and terms of use;
several require registration or a data-use agreement.

| source id        | city           | source                                         | reference                                                     | source-days |
|------------------|----------------|------------------------------------------------|---------------------------------------------------------------|-------------|
| `tdrive`         | Beijing        | T-Drive taxi trajectories                      | Yuan et al. (2010), doi:10.1145/1869790.1869807               | 7 |
| `geolife`        | Beijing        | GeoLife GPS trajectories                       | Zheng et al. (2009), doi:10.1145/1526709.1526816              | 8 |
| `jinan`          | Jinan          | Jinan taxi mobile road monitoring data         | Jinan Public Data Open Portal, http://data.jinan.gov.cn/jinan/catalog/1f3e4f65eebd4432ac046c2219a117d8 | 8 |
| `dublin`         | Dublin         | Google Project Air View, Dublin City           | https://data.smartdublin.ie/dataset/google-airview-data-dublin-city | 8 |
| `porto`          | Porto          | Taxi Service Trajectory (ECML PKDD 2015)       | UCI Machine Learning Repository, doi:10.24432/C55W25          | 8 |
| `montreal`       | Montreal       | MTL Trajet trip records                        | https://donnees.montreal.ca/dataset/mtl-trajet                | 8 |
| `barcelona`      | Barcelona      | Home-to-school pedestrian GPS (Beepath)        | Larroya et al. (2023), doi:10.1038/s41597-023-02328-3         | 8 |
| `rio`            | Rio de Janeiro | Bus mobility and environmental indicators      | Carvalho et al. (2025), doi:10.1038/s41597-025-05755-6        | 8 |
| `stellenbosch`   | Stellenbosch   | 1 Hz GPS tracking data (minibus)               | Hull (2024), doi:10.17632/xt69cnwh56.3                        | 4 |
| `nyc_mat`        | New York City  | Semantically enriched mobility dataset (MAT)   | Lettich et al. (2025), doi:10.5281/zenodo.15658129            | 8 |
| `paris_mat`      | Paris          | Semantically enriched mobility dataset (MAT)   | Lettich et al. (2025), doi:10.5281/zenodo.15658129            | 8 |
| `sf_cabspotting` | San Francisco  | CRAWDAD epfl/mobility taxi traces              | Piorkowski et al. (2009), doi:10.15783/C7J010                 | 8 |
| `rome_taxi`      | Rome           | CRAWDAD roma/taxi taxi traces                  | Bracciale et al. (2014), doi:10.15783/C7QC7M                  | 8 |
| `brighton_shl`   | Brighton       | Sussex-Huawei Locomotion dataset               | Gjoreski et al. (2018), doi:10.1109/ACCESS.2018.2858933       | 8 |

Total: 107 source-days. Licences and terms of use are those stated by each
provider; check them before obtaining or reusing the data.

## Monitoring sources used in the reconstruction tasks

The held-out monitoring tasks of the published study use Jinan and Dublin from
the table above and two further mobile-monitoring sources that are not part of
the access cohort:

| source              | measured quantity              | reference                                             |
|---------------------|--------------------------------|-------------------------------------------------------|
| Zurich OpenSense    | ultrafine particles, tram-mounted sensors | Maag et al. (2018), doi:10.5281/zenodo.1415369 |
| Portici AirHeritage | PM2.5, portable monitors       | De Vito et al. (2024), doi:10.5281/zenodo.13151961    |

## Preparing records

For each source-day the records must be

1. projected to a metric CRS appropriate for the city (metres);
2. clipped to one registered index day of the source's own qualified clock,
   with `t` in seconds since the start of that day, `0 <= t < 86400`;
3. written as flat `qrq.records.POINT` arrays with an int64 entity proxy and
   a strictly increasing `source_ordinal`.

The exact cleaning, projection and study-domain rules of the published run,
and the city reference points `(cx, cy)` used by `register_windows`, are
part of the restricted inputs described in `docs/reproducing.md`.
