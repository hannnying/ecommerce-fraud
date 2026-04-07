def orm_to_entity(orm_obj, entity_cls):
    field_names = entity_cls.__dataclass_fields__.keys()

    return entity_cls(
        **{
            name: getattr(orm_obj, name) for name in field_names if hasattr(orm_obj, name)
        }
    )

def entity_to_orm(orm_obj, entity_cls):
    field_names = orm_obj.__table__.columns

    return orm_obj(
        **{
            name: getattr(entity_cls, name) for name in field_names if hasattr(entity_cls, name)
        }
    )