import itertools
import unittest

from app.agents.candidate_ranking import (
    alias_map,
    bayesian_adjusted_rating,
    provider_rank_score,
    rating_confidence,
    resolve_aliases,
    shortlist,
)
from app.models.multi_agent import CandidateObservation, CandidateRegistry, RegistryEntity
from app.models.schemas import TripRequest


def request():
    return TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                       travel_days=1, transportation="Transit", accommodation="Mid-range hotel",
                       preferences=["Museums"])


class CandidateRankingTests(unittest.TestCase):
    def test_rating_and_rank_normalization(self):
        adjusted = bayesian_adjusted_rating(4.8, 100)
        self.assertGreater(adjusted, 4.2)
        self.assertLessEqual(rating_confidence(5.0, 100000), 1.0)
        self.assertEqual(rating_confidence(4.9, None), 0.0)
        self.assertEqual(provider_rank_score(1), 1.0)
        self.assertAlmostEqual(provider_rank_score(4), 4 / 7)
        self.assertEqual(provider_rank_score(8), 0.0)
        self.assertEqual(provider_rank_score(9), 0.0)

    def test_canonical_precedence_is_independent_of_arrival_order(self):
        observations = [
            CandidateObservation(source_type="supplemental", normalized_query="z", query_index=2,
                                 provider_rank=1, provider_id="p1", name="Supplemental", address="S"),
            CandidateObservation(source_type="base_anchor", normalized_query="anchor", query_index=0,
                                 provider_rank=2, provider_id="p1", name="Base", address="B"),
            CandidateObservation(source_type="place_details", normalized_query="detail", query_index=0,
                                 provider_rank=1, provider_id="p1", name="Details", address=""),
        ]
        outcomes = set()
        for permutation in itertools.permutations(observations):
            registry = CandidateRegistry(run_id="run")
            for observation in permutation:
                registry.add(RegistryEntity(source_id="attraction:p1", provider_id="p1",
                                             entity_type="attraction", name=observation.name,
                                             address=observation.address, registered_by="experience",
                                             observations=[observation]), actor="experience")
            entity = registry.entities["attraction:p1"]
            outcomes.add((entity.name, entity.address, tuple(entity.query_provenance)))
        self.assertEqual(outcomes, {("Details", "B", ("anchor", "detail", "z"))})

    def test_shortlist_and_aliases_are_stable_for_registry_insertion_permutations(self):
        base = [
            RegistryEntity(source_id=f"attraction:p{index}", provider_id=f"p{index}",
                           entity_type="attraction", name=f"Museum {index}", rating=4.5,
                           user_rating_count=100, best_provider_rank=index,
                           query_provenance=["museum"], registered_by="experience")
            for index in range(1, 5)
        ]
        orders = []
        for values in (base, list(reversed(base))):
            registry = CandidateRegistry(run_id="run")
            for entity in values:
                registry.add(entity, actor="experience")
            ranked = shortlist(registry, "attraction", request(), limit=4)
            aliases = alias_map(ranked, "A")
            orders.append((list(aliases), list(aliases.values())))
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(resolve_aliases(["A1", "A2"], dict(zip(*orders[0])), expected_prefix="A"),
                         orders[0][1][:2])
        with self.assertRaises(ValueError):
            resolve_aliases(["A1", "A1"], dict(zip(*orders[0])), expected_prefix="A")


if __name__ == "__main__":
    unittest.main()
