"""The execution paths a campaign can run a tool through.

`local`    -- the tool's own interpreter, no HTTP (see local.py)
`loopback` -- the HTTP API on this machine (see remote.py)
`lan`      -- the same HTTP API from another machine (see remote.py; it is the
              same client, pointed at a non-loopback base URL)
"""
