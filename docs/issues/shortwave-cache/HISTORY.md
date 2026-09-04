# History of the SPEEDY shortwave-tendency gating bug in jax-gcm (`jcm`)

## Summary

Installed `jcm` 2.0.1 computes SPEEDY shortwave radiation fluxes every
timestep (`shortwave_rad_fluxes`), but only *returns* the resulting
temperature tendency on steps where `physics_data.shortwave_rad.compute_shortwave`
is true (every `nstrad = 3` steps, set by `set_physics_flags` /
`SpeedyFlags`). On the other 2 of every 3 steps it returns an all-zero
tendency instead. Fortran SPEEDY (`physics.f90`) also only *recomputes*
shortwave every `nstrad` steps, but it stores the heating rate (`tt_rsw`)
and re-adds it to the total tendency **every** step. jax-gcm never
re-applies (or caches) it — so on 2/3 of steps the model receives zero
shortwave heating, silently losing ~2/3 of the shortwave-driven warming
and unbalancing the TOA/surface energy budget over long integrations.

This is confirmed present in the installed package, in the tagged
"paper" release (v1.1.1, GMD 2026), and in the current tip of both
`main` and `dev` on GitHub as of 2026-09-04.

- Repository: **https://github.com/climate-analytics-lab/jax-gcm**
  (PyPI package name `jcm`; default branch `dev`). Public, `gh auth
  status` confirmed read access. Cloned with full history/tags to
  `/home/dbalwada/jaxESM/external/jax-gcm`.
- Installed package (`.../site-packages/jcm/`, version 2.0.1) is
  **byte-identical** to tag `v2.0.1`
  (`06fd4771bccf6f8df3a1f032ca0c6e4e7957cf2d`, 2026-06-13) for both
  `jcm/physics/radiation/speedy_shortwave.py` and
  `jcm/physics/speedy/speedy_terms.py` — confirmed with `diff`.
- Paper version: `v1.1.0` = `7fe2c48699587e42cf10179d1e26809f60cb8776`
  (2026-03-04), `v1.1.1` = `3609a953015b7cf995ad07c6d0cd99d2ae49a803`
  (2026-03-04, "Bump version number").

## The bug, as it stands today

`jcm/physics/radiation/speedy_shortwave.py`:

```python
@jit
def get_shortwave_rad_fluxes(state, physics_data, parameters, forcing, terrain):
    # if compute_shortwave is true, then compute shortwave radiation
    # otherwise return the same physics_data and empty tendencies
    zero_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)
    state, physics_data, parameters, forcing, terrain, tendencies = shortwave_rad_fluxes(
        (state, physics_data, parameters, forcing, terrain, zero_tendencies))
    return jax.lax.cond(
        physics_data.shortwave_rad.compute_shortwave,
        lambda: tendencies, lambda: zero_tendencies,
    ), physics_data
```

`shortwave_rad_fluxes` (the un-gated inner function) is called every
step and always recomputes `dfabs` → `ttend_swr` (physics.f90:160-162
logic) and stores flux diagnostics into `physics_data`, but the
`lax.cond` at the end of the wrapper discards the tendency itself
(`zero_tendencies`) whenever `compute_shortwave` is false. Nothing
downstream re-applies a cached value of `ttend_swr` on those steps.
`jcm/physics/speedy/speedy_terms.py::set_physics_flags` sets
`compute_shortwave = (step % nstrad == 0)` with `nstrad = 3`
(`jcm/physics/speedy/physical_constants.py:48`), so heating is applied
on 1 of every 3 steps only, i.e. the model receives roughly 1/3 of the
shortwave heating it should over the course of a run.

The identical pattern gates `get_clouds` too (cloud diagnostics don't
produce a tendency, so that part is harmless), and this whole
"gate-with-lax.cond-and-drop" pattern is verbatim identical on
`origin/main`, `origin/dev`, `v2.0.1`, and `v1.1.1`.

By contrast, the *other* radiation backends in the same codebase do it
correctly: `jcm/physics/radiation/__init__.py` provides
`cached_radiation_tendency()` / `rescale_cached_radiation()` helpers,
and `grey_two_stream/radiation_scheme.py`, `nn_emulator_scheme.py`, and
`rrtmgp.py` all call `cached_radiation_tendency(...)` to re-emit the
last computed SW+LW heating on non-compute steps. `speedy_shortwave.py`
was never wired up to this shared mechanism — it's a SPEEDY-specific
gap.

## Full history of the gating logic

The gating logic went through three phases:

### Phase 1 — introduced, then reverted within the same PR (Oct 2024, never released)

- **PR #74** "Implement the speedy.f90 logic that runs between calls to
  the various subroutines" (author `jvmadan` / J. Varan Madan, merged
  2024-11-01T22:50:34Z, `merge_commit_sha`
  `23118429e653d50efe70e1e5120d255a39e8e459`, base `main`).
  - Commit `ab992388649f25d6bb99b1c7713a3affea335694` ("Enable longer
    shortwave timestep", 2024-10-31) first introduced substep gating:

    ```diff
    --- a/jcm/shortwave_radiation.py
    +++ b/jcm/shortwave_radiation.py
    @@ -22,6 +22,9 @@ def get_shortwave_rad_fluxes(state: PhysicsState, physics_data: PhysicsData):
         dfabs(ix,il,kx) # Flux of short-wave radiation absorbed in each atmospheric layer
         '''
     
    +    if physics_data.DateData.model_steps % nstrad != 1:
    +        return PhysicsTendency(jnp.zeros_like(state.u_wind), jnp.zeros_like(state.v_wind),
    +                                jnp.zeros_like(state.temperature), jnp.zeros_like(state.specific_humidity)), physics_data
    +
         ix, il, kx = state.temperature.shape
    ```
    Same zero-tendency-on-skip pattern, no caching.
  - But commit `88658322222f92dacaad34483a49dff9cba801c2` ("Remove
    compute_shortwave logic for now", 2024-10-31, same PR branch)
    **reverted it** before the PR merged, restoring every-step
    computation. This revert is what actually landed on `main`, so
    `v0.1` (`287e6af7`, 2025-03-18) and `v1.0.0`'s early history are
    **not** affected — shortwave ran every step at that point.

### Phase 2 — the bug that stuck (May 2025) → still present in v1.1.1 and current main/dev

- **PR #151** "Fix shortwave timestepping" (author `eldavenport` /
  Ellen Davenport, opened 2025-05-13T20:44:49Z, **merged
  2025-05-21T00:36:26Z**, `merge_commit_sha`
  `91855031a0331049102c14222a876688fd3a35ae`, base `main`, closes issue
  **#77** "Revisit shortwave timestepping once model time tracking is
  finalized"). PR description: *"Adding functionality to only call
  clouds and shortwave radiation every 3 time steps (matches
  speedy.f90). This adds a function in the model that sets flags to
  run or not run various tendencies at each time step."*

  This PR reintroduced the gate, this time using `jax.lax.cond` (for
  jittability) instead of a Python `if`/`%`, reproducing the same
  drop-the-tendency behavior — and this is the version that survived:

  ```diff
  --- a/jcm/shortwave_radiation.py
  +++ b/jcm/shortwave_radiation.py
  @@ -17,6 +18,21 @@ def get_shortwave_rad_fluxes(
       boundaries: BoundaryData,
       geometry: Geometry
   ) -> tuple[PhysicsTendency, PhysicsData]:
  +
  +    # if compute_shortwave is true, then compute shortwave radiation
  +    # otherwise return the same physics_data and empty tendencies
  +    tendencies = PhysicsTendency.zeros(shape=state.temperature.shape)
  +    state, physics_data, parameters, boundaries, geometry, tendencies = jax.lax.cond(
  +        physics_data.shortwave_rad.compute_shortwave,
  +        shortwave_rad_fluxes,
  +        pass_fn,
  +        operand=(state, physics_data, parameters, boundaries, geometry, tendencies)
  +    )
  +
  +    return tendencies, physics_data
  ```

  (Note: this early form of the commit even skips the *recomputation*
  itself via `pass_fn` on non-radiation steps — later refactors, e.g.
  the code now installed, always recompute `shortwave_rad_fluxes` every
  step but still discard the tendency via `lax.cond` at the return —
  same net effect: no heating applied on 2/3 of steps, just slightly
  more/less redundant compute depending on version.)

  **This is the first commit whose bug reached a tagged release: it
  predates and is an ancestor of `v1.0.0`, `v1.1.0`, `v1.1.1` (the GMD
  2026 paper version), and every later tag through `v2.0.1`
  (installed).**

- Issue **#77** (closed alongside PR #151, author `jvmadan`, assignee
  `eldavenport`) only discussed *how* to implement the gate (JAX-traced
  boolean mask vs. Python-level `if` at trace time) — it never raised
  re-applying/caching the stored heating rate on skipped steps, so the
  physics gap was not on anyone's radar.

### Phase 3 — related follow-ups that did NOT fix the tendency-caching gap

- **PR #177** "Set shortwave code to run at step 3n rather than 3n+1"
  (merged 2025-07-08) — realigns *which* steps are compute steps, not
  whether skipped steps get heating.
- **PR #196** "Ensure physics_data shortwave fields are still
  populated when compute_shortwave is false" (merged 2025-08-03) —
  fixes relative humidity blow-up caused by `physics_data.shortwave_rad`
  *diagnostic fields* (not the temperature tendency) going stale/zero
  between compute steps. Good context: the author explicitly frames it
  as making "code logic more in line with speedy's" persistence
  behavior, but the fix is scoped to diagnostic fields consumed
  elsewhere, not to the `PhysicsTendency` returned by
  `get_shortwave_rad_fluxes`.
- **PR #167** "Make sure compute_shortwave is called in post_process
  if it would be called in the normal physics step" — opened, **never
  merged** (closed with no merge commit).
- Issue **#571** "Propagate f32-stable radiation idioms to
  grey_two_stream and SPEEDY radiation" (open) — float32 numerical
  precision hygiene, unrelated to this bug.
- Issue **#671** "Cached SW heating and surface SW flux are replayed
  for the whole radiation interval with no zenith rescaling" (closed)
  — this is about the *other* (grey_two_stream / nn_emulator / RRTMGP)
  radiation backends, which already have a working
  `cached_radiation_tendency` mechanism; the issue was about rescaling
  the cached value for the changing solar zenith angle between compute
  steps, not about SPEEDY lacking a cache entirely.
- Issue **#395** "seasonal pattern of surface net short-wave flux is
  reversed after 100 years" (closed) — a *different*, calendar-drift
  bug in long coupled runs; not this issue.

No open issue or PR in the repository currently describes "SPEEDY
shortwave/clouds tendency dropped on non-radiation steps" / "shortwave
heating not applied every step" / missing `tt_rsw`-style caching.
Searches for `shortwave`, `nstrad`, `compute_shortwave`, `tt_rsw`,
`energy budget`, "radiation substep", "every step radiation" turned up
nothing matching. **This appears to be a real, currently-unreported
bug.**

## Current upstream status (checked 2026-09-04)

- `origin/main` and `origin/dev` (default branch, tip
  `226172e4b4680b4eed6c40b182668eebc2d89368`, 2026-09-02): bug present,
  identical `lax.cond(compute_shortwave, lambda: tendencies, lambda:
  zero_tendencies)` pattern in `jcm/physics/radiation/speedy_shortwave.py`
  for both `get_shortwave_rad_fluxes` and `get_clouds`.
- `jcm/physics/radiation/__init__.py` on `dev` already provides
  `cached_radiation_tendency()` / `rescale_cached_radiation()`, used by
  `grey_two_stream`, `nn_emulator_scheme.py`, and `rrtmgp.py` — the fix
  pattern exists in-repo, it's just not applied to
  `speedy_shortwave.py`.
- Not fixed at any point between the paper's v1.1.1 tag and the
  current tip.

## Commit/tag reference table

| Ref | SHA | Date | Note |
|---|---|---|---|
| PR #74 (introduced, then reverted same PR) | `ab992388649f25d6bb99b1c7713a3affea335694` → reverted by `88658322222f92dacaad34483a49dff9cba801c2` | 2024-10-31 | Never released |
| v0.1 | `287e6af774ed872a7b6caca4b58aadc2fec57a50` | 2025-03-18 | No gating — shortwave every step |
| **PR #151 — bug lands for good** | `91855031a0331049102c14222a876688fd3a35ae` | **merged 2025-05-21T00:36:26Z** | Ancestor of all releases below |
| v1.0.0 | `5a6d1726d7d849cbb67af9575d88f0a30fe7f359` | 2025-12-12 | Bug present |
| v1.1.0 | `7fe2c48699587e42cf10179d1e26809f60cb8776` | 2026-03-04 | Bug present |
| v1.1.1 (GMD 2026 paper version) | `3609a953015b7cf995ad07c6d0cd99d2ae49a803` | 2026-03-04 | Bug present |
| v2.0.1 (installed `jcm==2.0.1`) | `06fd4771bccf6f8df3a1f032ca0c6e4e7957cf2d` | 2026-06-13 | Bug present, byte-identical to installed package |
| origin/dev (default branch tip) | `226172e4b4680b4eed6c40b182668eebc2d89368` | 2026-09-02 | Bug still present |

## Fix direction (not implemented here)

Wire `speedy_shortwave.py` up to the same
`cached_radiation_tendency` / cache-and-rescale pattern already used by
`grey_two_stream`, `nn_emulator_scheme.py`, and `rrtmgp.py`
(`jcm/physics/radiation/__init__.py`): store `ttend_swr` (and the flux
diagnostics currently already persisted) in `physics_data.shortwave_rad`
on compute steps, and on non-compute steps return the *cached* tendency
instead of `zero_tendencies` — mirroring Fortran SPEEDY's `tt_rsw`
storage/reapplication in `physics.f90`. `PhysicsData.shortwave_rad`
already carries `rsns`/`ftop`/`dfabs`/`rsds` across steps (see PR #196),
so the underlying storage exists; only the tendency return path needs
to stop discarding the temperature increment.
