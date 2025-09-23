import toml
from utils import (
    get_db_connection,
    init_scene_tables,
    clear_table,
    init_window_tables,
    iter_scenes,
)


def make_scene_chunks():
    """Process script chunks and insert them row-by-row into the database."""

    init_scene_tables("scene")

    # Clear existing data
    clear_table("scene")

    con = get_db_connection()
    cur = con.cursor()

    table_name = "scene"

    try:
        # Process all files in the scripts directory
        # for file_name in os.listdir("scripts"):
        for file_name in [
            "4x12 A New Man.txt",
        ]:
            if not file_name.endswith(".txt"):
                continue

            print(f"Splitting file: {file_name}")

            with open(f"scripts/{file_name}", "r") as f:
                script = f.read()

            # Split on lines that start with "cut to" (case insensitive)
            lines = script.split("\n")

            # Initialize variables.
            # curr_chunk = a list of individual lines for the current scene. Gets reset each time a "cut to" is encountered.
            curr_chunk = []  # current scene chunk
            chunk_index = 0

            for line in lines:
                # If we've hit a new scene, start a new chunk.
                if line.strip().lower().startswith(
                    "cut to"
                ) | line.strip().lower().startswith("(cut to"):
                    # If we have accumulated lines, save as a chunk and insert into DB
                    if curr_chunk:
                        chunk_text = "\n".join(curr_chunk).strip()
                        if chunk_text:
                            print(f"  Splitting chunk {chunk_index}")

                            # Insert into main table
                            cur.execute(
                                f"""
                                INSERT INTO {table_name}
                                (file_name, scene_id_in_episode, scene_text)
                                VALUES (?, ?, ?)
                            """,
                                (
                                    file_name,
                                    chunk_index,
                                    chunk_text,
                                ),
                            )
                            chunk_index += 1

                    # Reset current_scene_chunk to be this first line of the new scene
                    curr_chunk = [line]

                # If the new line is not a new scene, we append the line to the current scene list.
                else:
                    curr_chunk.append(line)

            # Add the final chunk if it exists
            if curr_chunk:
                chunk_text = "\n".join(curr_chunk).strip()
                if chunk_text:
                    print(f"  Processing chunk {chunk_index}")

                    # Insert into main table
                    cur.execute(
                        f"""
                        INSERT INTO {table_name}
                        (file_name, scene_id_in_episode, scene_text)
                        VALUES (?, ?, ?)
                    """,
                        (
                            file_name,
                            chunk_index,
                            chunk_text,
                        ),
                    )

        con.commit()
        print(f"Successfully inserted scene chunks into table: `{table_name}`")

    except Exception as e:
        print(f"Error in make_scene_chunks: {e}")
    finally:
        con.close()


def make_window_chunk(chunk):
    """Create window chunks from a text chunk."""
    # Load window configuration
    config = toml.load("config.toml")
    window_size = config["WINDOW"]["window_size"]
    step_size = config["WINDOW"]["step_size"]

    # Split script into "formatted lines" (by double newlines)
    # This keeps speaker names with their dialogue
    formatted_lines = chunk.split("\n\n")

    # Remove empty formatted lines
    formatted_lines = [line.strip() for line in formatted_lines if line.strip()]

    script_chunks = []
    i = 0

    while i < len(formatted_lines):
        # Create window of formatted lines
        window_end = min(i + window_size, len(formatted_lines))
        window_formatted_lines = formatted_lines[i:window_end]

        # Join the formatted lines back with double newlines
        chunk_text = "\n\n".join(window_formatted_lines).strip()
        if chunk_text:
            script_chunks.append(chunk_text)

        # Move window forward by step_size
        i += step_size

        # Break if we've reached the end
        if window_end >= len(formatted_lines):
            break

    return script_chunks


def iter_windows_from_scenes():
    """Read each scene, make windows from them."""

    for scene in iter_scenes():
        windows = make_window_chunk(scene["text"])

        for w_idx, w_text in enumerate(windows):
            yield {
                "scene_id": scene["scene_id"],
                "window_id_in_scene": w_idx,
                "window_text": w_text,
                "file_name": scene["file_name"],
            }


def insert_window_db():
    """Insert window chunks into the database."""

    init_window_tables("window")
    clear_table("window")

    con = get_db_connection()
    cur = con.cursor()

    table_name = "window"
    try:
        for row in iter_windows_from_scenes():
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (scene_id, window_id_in_scene, window_text, file_name)
                VALUES (?, ?, ?, ?)
            """,
                (
                    row["scene_id"],
                    row["window_id_in_scene"],
                    row["window_text"],
                    row["file_name"],
                ),
            )

        con.commit()
        print(f"Successfully inserted window chunks into table: `{table_name}`")

    except Exception as e:
        print(f"Error in insert_window_db: {e}")
    finally:
        con.close()
