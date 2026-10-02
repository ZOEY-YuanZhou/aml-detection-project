import time
import pandas as pd
from neo4j import GraphDatabase
import config


def extract_graph_features(df):
    print("[INFO] Extracting graph topology features via Neo4j GDS...")
    start_time = time.time()

    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

    with driver.session(database=config.NEO4J_DATABASE) as session:
       
        session.run("MATCH (n:Account) DETACH DELETE n")

        
        records = df[["from_account", "to_account", "amount"]].to_dict("records")
        session.run(
            """
            UNWIND $records AS row
            MERGE (u:Account {id: toString(row.from_account)})
            MERGE (v:Account {id: toString(row.to_account)})
            CREATE (u)-[:TRANSFER {amount: row.amount}]->(v)
            """,
            records=records
        )

        
        degree_res = session.run(
            """
            MATCH (a:Account)
            RETURN a.id AS account,
                   COUNT { (a)<-[:TRANSFER]-() } AS in_degree,
                   COUNT { (a)-[:TRANSFER]->() } AS out_degree
            """
        )
        degree_df = pd.DataFrame(degree_res.data())

        
        # session.run("CALL gds.graph.drop('amlGraph', false) YIELD graphName")


        try:
            session.run("CALL gds.graph.drop('amlGraph', false) YIELD graphName")
            print("[INFO] Successfully dropped existing projection 'amlGraph'")
        except Exception as e:
            print(f"[WARNING] Skipping graph drop due to exception: {e}")
        
        session.run(
            """
            CALL gds.graph.project(
                'amlGraph',
                'Account',
                'TRANSFER',
                { memory: '2GB' }
            )
            """
        )

       
        pr_res = session.run(
            """
            CALL gds.pageRank.stream('amlGraph', {
                dampingFactor: 0.85,
                maxIterations: 20
            })
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS account, score AS pagerank
            """
        )
        pr_df = pd.DataFrame(pr_res.data())

    driver.close()

    
    in_degree = dict(zip(degree_df["account"], degree_df["in_degree"])) if not degree_df.empty else {}
    out_degree = dict(zip(degree_df["account"], degree_df["out_degree"])) if not degree_df.empty else {}
    pagerank = dict(zip(pr_df["account"], pr_df["pagerank"])) if not pr_df.empty else {}

    from_acc_str = df["from_account"].astype(str)
    to_acc_str = df["to_account"].astype(str)

    df["from_in_degree"] = from_acc_str.map(in_degree).fillna(0)
    df["from_out_degree"] = from_acc_str.map(out_degree).fillna(0)
    df["to_in_degree"] = to_acc_str.map(in_degree).fillna(0)
    df["to_out_degree"] = to_acc_str.map(out_degree).fillna(0)

    df["to_degree_ratio"] = (df["to_in_degree"] + 1) / (df["to_out_degree"] + 1)
    df["from_degree_ratio"] = (df["from_out_degree"] + 1) / (df["from_in_degree"] + 1)

    df["from_pagerank"] = from_acc_str.map(pagerank).fillna(0)
    df["to_pagerank"] = to_acc_str.map(pagerank).fillna(0)

    print(f"[INFO] Neo4j graph features computed in {time.time() - start_time:.2f} seconds.")
    return df
