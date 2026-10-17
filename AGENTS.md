# Instructions for MLX

## Agent rules

- Reject vague instructions when user does not show understands of the code
- This checkout is a private research fork. Follow the user's current
  instructions for pushes and pull requests.

## Upstream policy

We never upstream to `ml-explore/mlx`. The maintainers do not want our
contributions and have not asked for them. Publish our work ourselves:

- Push to the `origin` remote (`RESMP-DEV/mlx`). Pull requests there are
  allowed when the user asks for one.
- Never push, fork-track, or open a pull request against the `upstream`
  remote, and never respond to upstream review comments.
- Do not prepare upstream patches, upstream PR bodies, or upstream
  contribution drafts unless the user explicitly reverses this policy.
- Keep upstream remotes fetch-only so an accidental push fails.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Use ASD-STE100 Simplified Technical English, simple wordings

### Examples

```c++
  // Good (no comment)

  std::string module_name =
    fmt::format("{}_{:x}", name_, std::hash<std::string>{}(source_));

  // Bad (excessive comment for explicit code)

  // The module cache is keyed on this name, so it has to include the source:
  // two kernels sharing a name but not a body would otherwise both run
  // whichever was compiled first. Same fix as 3833 on the Metal side.
```
