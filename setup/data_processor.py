import toml
import os
from utils.database import get_db_connection
from utils.data_access import iter_scenes


def make_scene_chunks():
    """Process script chunks and insert them row-by-row into the database."""

    con = get_db_connection()
    cur = con.cursor()

    table_name = "scene"

    config = toml.load("config.toml")
    scene_split_markers = config["WINDOW"]["scene_split_markers"]
    try:
        # Process all files in the scripts directory
        for file_name in os.listdir("scripts"):
            # for file_name in [
            #     "1x01 Welcome to the Hellmouth.txt",
            #     "4x12 A New Man.txt",
            # ]:
            if not file_name.endswith(".txt"):
                continue

            print(f"Splitting file: {file_name}")

            with open(f"scripts/{file_name}", "r") as f:
                script = f.read()

            # Split on lines that start with "cut" (case insensitive)
            lines = script.split("\n")

            # Initialize variables.
            # curr_chunk = a list of individual lines for the current scene. Gets reset each time a "cut" is encountered.
            curr_chunk = []  # current scene chunk
            chunk_index = 0
            episode_name = file_name.replace(".txt", "")

            for line in lines:
                # If we've hit a new scene, start a new chunk.
                line_lower = line.strip().lower()
                if any(line_lower.startswith(marker) for marker in scene_split_markers):
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
                                    episode_name,
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
                            episode_name,
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
    w_start = 0

    # slices of formatted lines
    slices = []

    while w_start < len(formatted_lines):
        # Create window of formatted lines
        w_end = min(w_start + window_size, len(formatted_lines))
        window_formatted_lines = formatted_lines[w_start:w_end]
        slices.append([w_start, w_end])

        # Join the formatted lines back with double newlines
        chunk_text = "\n\n".join(window_formatted_lines).strip()
        if chunk_text:
            script_chunks.append(chunk_text)

        # Move window forward by step_size
        w_start += step_size

        # Break if we've reached the end
        if w_end >= len(formatted_lines):
            break

    return script_chunks, slices


def iter_windows_from_scenes():
    """Read each scene, make windows from them."""

    for scene in iter_scenes():
        windows, slices = make_window_chunk(scene["text"])

        for w_idx, (w_text, slice) in enumerate(zip(windows, slices)):
            yield {
                "scene_id": scene["scene_id"],
                "window_id_in_scene": w_idx,
                "window_start": slice[0],
                "window_end": slice[1],
                "window_text": w_text,
                "file_name": scene["file_name"],
            }


def insert_window_db():
    """Insert window chunks into the database using batch processing."""

    con = get_db_connection()
    cur = con.cursor()

    table_name = "window"
    batch_size = 1000  # Process windows in batches
    batch = []
    total_processed = 0

    try:
        for row in iter_windows_from_scenes():
            batch.append(
                (
                    row["scene_id"],
                    row["window_id_in_scene"],
                    row["window_start"],
                    row["window_end"],
                    row["window_text"],
                    row["file_name"],
                )
            )

            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                cur.executemany(
                    f"""
                    INSERT INTO {table_name}
                    (scene_id, window_id_in_scene, window_start, window_end, window_text, file_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    batch,
                )
                con.commit()  # Commit each batch
                total_processed += len(batch)
                print(f"Processed {total_processed} window chunks...")
                batch = []  # Reset batch

        # Process any remaining windows in the final batch
        if batch:
            cur.executemany(
                f"""
                INSERT INTO {table_name}
                (scene_id, window_id_in_scene, window_start, window_end, window_text, file_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                batch,
            )
            con.commit()
            total_processed += len(batch)

        print(
            f"Successfully inserted {total_processed} window chunks into table: `{table_name}`"
        )

    except Exception as e:
        print(f"Error in insert_window_db: {e}")
        con.rollback()  # Rollback on error
    finally:
        con.close()
