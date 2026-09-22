-- Ejecutar una sola vez en Supabase: dashboard del proyecto -> SQL Editor ->
-- pegar esto -> Run. Crea la tabla donde vive la memoria de Mayordomo.

create table if not exists memoria_mayordomo (
  visitor_id text primary key,
  texto text not null default '',
  historial jsonb not null default '[]'::jsonb,
  actualizado timestamptz not null default now()
);

-- Si ya creaste la tabla antes de esta version (sin la columna historial),
-- corre esta linea aparte para agregarla sin perder lo que ya tenias:
-- alter table memoria_mayordomo add column if not exists historial jsonb not null default '[]'::jsonb;

-- mantiene "actualizado" al dia solo (no es obligatorio, pero ayuda si
-- alguna vez queres ver ultima actividad por persona)
create or replace function actualizar_fecha_memoria()
returns trigger as $$
begin
  new.actualizado = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_actualizar_fecha_memoria on memoria_mayordomo;
create trigger trg_actualizar_fecha_memoria
  before update on memoria_mayordomo
  for each row execute function actualizar_fecha_memoria();

-- Seguridad: esta tabla solo la toca el servidor de Mayordomo (con la
-- service_role key, que salta Row Level Security), nunca el navegador de
-- la persona directamente. Por eso no hace falta configurar politicas RLS
-- especiales; dejar RLS desactivado en esta tabla es suficiente y mas
-- simple. Si en algun momento se llegara a usar la key publica ("anon")
-- desde el navegador, ahi si habria que activar RLS con politicas por
-- visitor_id antes de exponerla.
