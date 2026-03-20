import fakeredis
import pytest
from src.state.device_state_flexible import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState

@pytest.fixture
def fake_redis():
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    yield client
    client.flushall()


@pytest.fixture
def device_state(fake_redis):
    state = DeviceState()
    state.client = fake_redis
    return state


@pytest.fixture
def global_velocity(fake_redis):
    state = GlobalVelocity()
    state.client = fake_redis
    return state


@pytest.fixture
def ip_state(fake_redis):
    state = IPState()
    state.client = fake_redis
    return state